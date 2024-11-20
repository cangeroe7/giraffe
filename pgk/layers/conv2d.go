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
	Padded     bool

	padding []int

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
  c.CompilePadding(inShape)

  // Initialize biases to zero: 1 per filter
	c.biases = t.ZerosTensor([]int{1, c.Filters})

  // Compute output shape
	outHeight := (c.padding[0]+c.padding[2]+inShape.Rows()-c.KernelSize[0])/c.Strides[0] + 1
	outWidth := (c.padding[1]+c.padding[3]+inShape.Cols()-c.KernelSize[0])/c.Strides[1] + 1

	var outShape t.Shape = []int{c.Filters, outHeight, outWidth}

  // Xavier/Glorot Initialization
	limit := math.Sqrt(6 / float64(inShape.TotalSize()) + float64(outShape.TotalSize()))


  // Initialize weights/kernels to random value between -limit and limit
	var err error
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

	return nil, nil
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

func (c *Conv2D) CompilePadding(shape t.Shape) {
  if !c.Padded {
    c.padding = make([]int, 4)
     return
  }
	inHeight := shape.Rows()

	outHeight := int(math.Ceil(float64(inHeight) / float64(c.Strides[0])))

	padHeight := ((outHeight-1)*c.Strides[0] + c.KernelSize[0] - inHeight) / 2

	var B, T, R, L int
	if padHeight > 0 {
		B = c.Strides[0] / 2
		T = c.Strides[0] - B
	} else {
		B = padHeight
		T = padHeight
	}

	inWidth := shape.Cols()

	outWidth := int(math.Ceil(float64(shape.Cols()) / float64(c.Strides[1])))

	padWidth := ((outWidth-1)*c.Strides[1] + c.KernelSize[1] - inWidth) / 2

	if padHeight > 0 {
		R = c.Strides[1] / 2
		L = c.Strides[1] - R
	} else {
		R = padWidth
		L = padWidth
	}

	c.padding = []int{T, R, B, L}
}
