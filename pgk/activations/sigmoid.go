package activations

import (
	"fmt"
	"math"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Sigmoid struct {
	input t.Tensor
}

func sigmoid() Activation {
  return &Sigmoid{}
}

func (a *Sigmoid) Type() string {
  return "sigmoid"
}

func (a *Sigmoid) Forward(input t.Tensor) (t.Tensor, error) {

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

func (a *Sigmoid) Backward(gradient t.Tensor) (t.Tensor, error) {

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
