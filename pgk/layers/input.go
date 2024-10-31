package layers

import (
	"errors"

	u "github.com/cangeroe7/giraffe/internal/utils"
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

func (i *input) Forward(input u.Matrix) (u.Matrix, error) {
	inShape := input.Shape()
	if len(inShape) != len(i.shape) {
		return nil, errors.New("Not the same number of dimensions")
	}

	if inShape[1] != i.shape[1] {
		return nil, errors.New("Dimensions do not match")
	}

	return input, nil
}

func (i *input) Backward(gradient u.Matrix) (u.Matrix, error) {
	return gradient, nil
}

func (i *input) CompileLayer(inShape []int) ([]int, error) {
	return inShape, nil
}

func (i *input) Weights() u.Matrix {
	return nil
}

func (i *input) Biases() u.Matrix {
	return nil
}

func (i *input) WeightsGradient() u.Matrix {
	return nil
}

func (i *input) BiasesGradient() u.Matrix {
	return nil
}
