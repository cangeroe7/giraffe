package layers

import (
	"errors"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type input struct {
	shape []int
}

func Input(shape []int) Layer {
	return &input{shape: shape}
}

func (i *input) Type() string {
	return "input"
}

func (i *input) Forward(input t.Tensor) (t.Tensor, error) {
	inShape := input.Shape()
	if len(inShape) != len(i.shape) {
		return nil, errors.New("Not the same number of dimensions")
	}

	if inShape[1] != i.shape[1] {
		return nil, errors.New("Dimensions do not match")
	}

	return input, nil
}

func (i *input) Backward(gradient t.Tensor) (t.Tensor, error) {
	return gradient, nil
}

func (i *input) CompileLayer(inShape []int) ([]int, error) {
	return inShape, nil
}

func (i *input) Weights() t.Tensor {
	return nil
}

func (i *input) Biases() t.Tensor {
	return nil
}

func (i *input) WeightsGradient() t.Tensor {
	return nil
}

func (i *input) BiasesGradient() t.Tensor {
	return nil
}
