package layers

import (
	"errors"
	"fmt"
	"math"

	a "github.com/cangeroe7/giraffe/pgk/activations"
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Conv2D struct {
	Filters    int
	KernelSize [2]int
	Strides    [2]int
	Mode       PaddingMode

	padding []int

	input t.Tensor

	inShape         t.Shape
	outShape        t.Shape
	weights         t.Tensor
	biases          t.Tensor
	weightsGradient t.Tensor
	biasesGradient  t.Tensor
	Activation      a.Activation
}

func (c *Conv2D) Type() string {
	return "Conv2D"
}

func (c *Conv2D) Params() map[string]interface{} {
	return map[string]interface{}{
		"filters":     c.Filters,
		"activation":  c.Activation.Type(),
		"kernel_size": c.KernelSize,
		"strides":     c.Strides,
		"mode":        c.Mode,
		"padding":     c.padding,
	}
}

func (c *Conv2D) CompileLayer(inShape t.Shape) (t.Shape, error) {
	if c.Filters <= 0 {
		return nil, errors.New("Must be 1 or more filters")
	}

	// Check kernel sizes and sets default if needed
	switch {
	case c.KernelSize[0] < 0 || c.KernelSize[1] < 0:
		return nil, errors.New("Negative kernel value")

	case (c.KernelSize[0] > 0) != (c.KernelSize[1] > 0):
		return nil, errors.New("One kernel set to zero, must be positive")

		// Default when no kernel is given
	case c.KernelSize[0] == 0 && c.KernelSize[1] == 0:
		c.KernelSize[0], c.KernelSize[1] = 1, 1
	}

	// Check stride sizes and sets default if needed
	switch {
	case c.Strides[0] < 0 || c.Strides[1] < 0:
		return nil, errors.New("Negative stride value")

	case (c.Strides[0] > 0) != (c.Strides[1] > 0):
		return nil, errors.New("One stride set to zero, must be positive")

	case c.Strides[0] == 0 && c.Strides[1] == 0:
		c.Strides[0], c.Strides[1] = 1, 1
	}

	// Set padding values
	padding, err := ComputePadding(inShape, c.KernelSize, c.Strides, c.Mode)
	if err != nil {
		return nil, err
	}

	c.padding = padding

	// Initialize biases to zero: 1 per filter
	c.biases = t.ZerosTensor([]int{1, c.Filters})

	// Compute output shape
	outHeight := (padding[0]+padding[2]+inShape.Rows()-c.KernelSize[0])/c.Strides[0] + 1
	outWidth := (padding[1]+padding[3]+inShape.Cols()-c.KernelSize[0])/c.Strides[1] + 1

	var outShape t.Shape = []int{c.Filters, outHeight, outWidth}

	// Xavier/Glorot Initialization

	limit := math.Sqrt(6 / float64(inShape.Channels()+c.Filters))

	// Initialize weights/kernels to random value between -limit and limit
	c.weights, err = t.RandTensor([]int{c.Filters, inShape.Channels(), c.KernelSize[0], c.KernelSize[1]}, -limit, limit)
	if err != nil {
		return nil, err
	}

	// Default activation function
	if c.Activation == nil {
		c.Activation = &a.Relu{}
	}

	return outShape, nil
}

func (c *Conv2D) Forward(input t.Tensor) (t.Tensor, error) {
	c.inShape = input.Shape().Clone()

	padInput, err := input.Pad(c.padding...)
	if err != nil {
		return nil, err
	}

	c.input = padInput

	outHeight := (padInput.Shape().Rows()-c.KernelSize[0])/c.Strides[0] + 1
	outWidth := (padInput.Shape().Cols()-c.KernelSize[1])/c.Strides[1] + 1

	resTen := t.ZerosTensor([]int{input.Shape().Batches(), c.Filters, outHeight, outWidth})

	inIter, err := t.IterFromTensor(padInput, "batches")
	if err != nil {
		return nil, err
	}

	resIter, err := t.IterFromTensor(resTen, "mat")
	if err != nil {
		return nil, err
	}

	for inBatch, ok := inIter.Next(); ok; inBatch, ok = inIter.Next() {

		filterIter, err := t.IterFromTensor(c.weights, "batches")
		if err != nil {
			return nil, err
		}

		currentBias := 0
		for filter, ok := filterIter.Next(); ok; filter, ok = filterIter.Next() {
			resMat, ok := resIter.Next()

			if !ok {
				return nil, errors.New("result Tensor doesn't have enough matrices for convolved input")
			}

			_, err := inBatch.CrossCorrelate(filter, c.Strides, resMat)
			if err != nil {
				return nil, err
			}

			// Add the bias
			resMat.ScalarAdd(c.biases.ValueAt(currentBias), true)
			currentBias++
		}
	}

	// Apply the activation function
	resTen, err = c.Activation.Forward(resTen)
	if err != nil {
		return nil, err
	}

	return resTen, nil
}

func (c *Conv2D) Backward(gradient t.Tensor) (t.Tensor, error) {

	gradient, err := c.Activation.Backward(gradient)
	if err != nil {
		return nil, err
	}

	if gradient.Shape().Channels() != c.Filters {
		return nil, errors.New("gradient shape does not match the output shape")
	}

	// Compute the biases gradient
	err = c.computeBiasGradient(gradient)
	if err != nil {
		return nil, err
	}

	// Compute the weights gradient
	err = c.computeWeightsGradient(gradient)
	if err != nil {
		return nil, err
	}

	// Compute the input gradient
	inputGradient, err := c.computeInputGradient(gradient)
	if err != nil {
		return nil, err
	}

	return inputGradient, nil
}

func (c *Conv2D) computeBiasGradient(gradient t.Tensor) error {
	c.biasesGradient = t.ZerosTensor([]int{1, c.Filters})

	// Compute the bias gradient: sum each output filter gradient
	gradientBatchIter, err := t.IterFromTensor(gradient, "batches")
	if err != nil {
		return err
	}

	for gradientBatch, ok := gradientBatchIter.Next(); ok; gradientBatch, ok = gradientBatchIter.Next() {

		gradientFilterIter, err := t.IterFromTensor(gradientBatch, "matrix")
		if err != nil {
			return err
		}

		i := 0
		for gradientFilter, ok := gradientFilterIter.Next(); ok; gradientFilter, ok = gradientFilterIter.Next() {
			c.biasesGradient.SetValueAt(i, c.biasesGradient.ValueAt(i)+gradientFilter.Sum())
			i++
		}
	}
	return nil
}

func (c *Conv2D) computeWeightsGradient(gradient t.Tensor) error {
	weightsGradient := t.ZerosTensor(c.weights.Shape().Clone())

	gradient, err := gradient.Dilate(c.Strides[0]-1, c.Strides[1]-1)
	if err != nil {
		return err
	}

	gradientBatchIter, err := t.IterFromTensor(gradient, "batches")
	inputBatchIter, err := t.IterFromTensor(c.input, "batches")

	for gradientBatch, ok := gradientBatchIter.Next(); ok; gradientBatch, ok = gradientBatchIter.Next() {
		inputBatch, _ := inputBatchIter.Next()

		filterGradientIter, err := t.IterFromTensor(weightsGradient, "channels")
		if err != nil {
			return err
		}

		gradientChannelIter, err := t.IterFromTensor(gradientBatch, "channels")

		for gradientChannel, ok := gradientChannelIter.Next(); ok; gradientChannel, ok = gradientChannelIter.Next() {

			inputChannelIter, err := t.IterFromTensor(inputBatch, "channels")
			if err != nil {
				return err
			}

			for inputChannel, ok := inputChannelIter.Next(); ok; inputChannel, ok = inputChannelIter.Next() {
				filterGradient, _ := filterGradientIter.Next()

				_, err := inputChannel.CrossCorrelate(gradientChannel, [2]int{1, 1}, filterGradient)
				if err != nil {
					fmt.Println("Error computing weights gradient crosss correlate")
					return err
				}
			}
		}
	}

	c.weightsGradient = weightsGradient
	return nil
}

func (c *Conv2D) computeInputGradient(gradient t.Tensor) (t.Tensor, error) {

	inputGradient := t.ZerosTensor(c.inShape)

	gradient, err := gradient.Dilate(c.Strides[0]-1, c.Strides[1]-1)

	gradient, err = gradient.Pad(c.KernelSize[0]-1, c.KernelSize[1]-1)
	if err != nil {
		return nil, err
	}

	gradientBatchIter, err := t.IterFromTensor(gradient, "batches")
	if err != nil {
		return nil, err
	}

	inputGradientBatchIter, err := t.IterFromTensor(inputGradient, "batches")

	for gradientBatch, ok := gradientBatchIter.Next(); ok; gradientBatch, ok = gradientBatchIter.Next() {
		inputGradientBatch, _ := inputGradientBatchIter.Next()

		filterIter, err := t.IterFromTensor(c.weights, "channels")
		if err != nil {
			return nil, err
		}

		gradientChannelIter, err := t.IterFromTensor(gradientBatch, "channel")
		if err != nil {
			return nil, err
		}

		for gradientChannel, ok := gradientChannelIter.Next(); ok; gradientChannel, ok = gradientChannelIter.Next() {
			inputGradientChannelIter, err := t.IterFromTensor(inputGradientBatch, "channels")
			if err != nil {
				return nil, err
			}

			for inputGradientChannel, ok := inputGradientChannelIter.Next(); ok; inputGradientChannel, ok = inputGradientChannelIter.Next() {
				filter, _ := filterIter.Next()
				_, err := gradientChannel.Convolve(filter, [2]int{1, 1}, inputGradientChannel)
				if err != nil {
					return nil, err
				}
			}
		}
	}

	inputGradient, err = inputGradient.Trim(c.padding...)
	if err != nil {
		return nil, err
	}

	return inputGradient, nil
}

func (c *Conv2D) Weights() t.Tensor {
	return c.weights
}

func (c *Conv2D) Biases() t.Tensor {
	return c.biases
}

func (c *Conv2D) WeightsGradient() t.Tensor {
	return c.weightsGradient
}

func (c *Conv2D) BiasesGradient() t.Tensor {
	return c.biasesGradient
}

func Conv2DFromParams(params map[string]interface{}, weights []float64, biases []float64) (Layer, error) {

	filtersFloat64, ok := params["filters"].(float64)
	if !ok {
		return nil, errors.New("missing or invalid 'filters' parameter")
	}

	filters := int(filtersFloat64)

	kernelSizeInterface, ok := params["kernel_size"].([]interface{})
	if !ok {
		return nil, errors.New("missing or invalid 'kernel_size' parameter")
	}

	kernelSized, err := interfaceToIntArray(kernelSizeInterface)
	if err != nil {
		return nil, err
	}

	kernelSize := [2]int{kernelSized[0], kernelSized[1]}

	stridesInterface, ok := params["strides"].([]interface{})
	if !ok {
		return nil, errors.New("missing or invalid 'kernel_size' parameter")
	}

	stride, err := interfaceToIntArray(stridesInterface)
	if err != nil {
		return nil, err
	}

	strides := [2]int{stride[0], stride[1]}

	paddingInterface, ok := params["padding"].([]interface{})
	if !ok {
		return nil, errors.New("missing or invalid 'padding' parameter")
	}

	padding, err := interfaceToIntArray(paddingInterface)
	if err != nil {
		return nil, err
	}

	activation, ok := params["activation"].(string)
	if !ok {
		return nil, errors.New("missing or invalid 'activation' parameter")
	}

	mode, ok := params["mode"].(string)
	if !ok {
		return nil, errors.New("missing or invalid 'mode' parameter")
	}

	activationStruct := a.Activations[activation]()

	biasesShape := []int{1, filters}

	inChannels := len(weights) / (filters * kernelSize[0] * kernelSize[1])
	weightsShape := []int{filters, inChannels, kernelSize[0], kernelSize[1]}

	weightsTensor, err := t.TensorFrom(weightsShape, weights)
	if err != nil {
		return nil, err
	}

	biasesTensor, err := t.TensorFrom(biasesShape, biases)
	if err != nil {
		return nil, err
	}

	return &Conv2D{
		Filters:    filters,
		KernelSize: kernelSize,
		Strides:    strides,
		Activation: activationStruct,
		Mode:       PaddingMode(mode),
		padding:    padding,
		weights:    weightsTensor,
		biases:     biasesTensor,
	}, nil
}
