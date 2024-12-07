package layers

import (
	"errors"
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
	outHeight := (c.padding[0]+c.padding[2]+inShape.Rows()-c.KernelSize[0])/c.Strides[0] + 1
	outWidth := (c.padding[1]+c.padding[3]+inShape.Cols()-c.KernelSize[0])/c.Strides[1] + 1

	var outShape t.Shape = []int{c.Filters, outHeight, outWidth}

	// Xavier/Glorot Initialization
	limit := math.Sqrt(6/float64(inShape.TotalSize()) + float64(outShape.TotalSize()))

	// Initialize weights/kernels to random value between -limit and limit
	c.weights, err = t.RandTensor([]int{c.Filters, inShape.Channels(), c.KernelSize[0], c.KernelSize[1]}, -limit, limit)
	if err != nil {
		return nil, err
	}

	// Default activation function
	if c.Activation == nil {
		c.Activation = a.Relu()
	}

	return outShape, nil
}

func (c *Conv2D) Forward(input t.Tensor) (t.Tensor, error) {

	padInput, err := input.Pad(c.padding...)
	if err != nil {
		return nil, err
	}

	outHeight := (padInput.Shape().Rows()-c.KernelSize[0])/c.Strides[0] + 1
	outWidth := (padInput.Shape().Cols()-c.KernelSize[0])/c.Strides[1] + 1

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
	// Initialize gradients for weights and biases
	c.weightsGradient = t.ZerosTensor(c.weights.Shape())

	return nil, nil
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
  c.weightsGradient = t.ZerosTensor(c.weights.Shape().Clone())

  
  
  return nil
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
