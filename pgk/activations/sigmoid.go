package activations

import (
	"fmt"
	"math"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

func Sigmoid() Activation { return &sigmoid{} }

type sigmoid struct {
	input t.Tensor
}

func (a *sigmoid) Type() string {
  return "sigmoid"
}

func (a *sigmoid) Forward(input t.Tensor) (t.Tensor, error) {

	a.input = input

	sigmoid := func(x float64) (float64, error) {
		output := 1 / (1 + math.Exp(-x))
		return output, nil
	}

	output, err := input.Map(sigmoid, false)
	if err != nil {
		return nil, err
	}

	return output, nil
}

func (a *sigmoid) Backward(gradient t.Tensor) (t.Tensor, error) {

	sigmoidPrime := func(x float64) (float64, error) {
		sigmoid := 1 / (1 + math.Exp(-x))
		output := sigmoid * (1 - sigmoid)
		return output, nil
	}

	primeInput, err := a.input.Map(sigmoidPrime, false)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return nil, err
	}

	outGradient, err := gradient.Multiply(primeInput, false)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return nil, err
	}

	return outGradient, nil
}
