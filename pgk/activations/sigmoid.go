package activations

import (
	"fmt"
	"math"

	u "github.com/cangeroe7/giraffe/internal/utils"
)

func Sigmoid() Activation { return &sigmoid{} }

type sigmoid struct {
	input u.Matrix
}

func (a *sigmoid) Type() string {
  return "sigmoid"
}

func (a *sigmoid) Forward(input u.Matrix) (u.Matrix, error) {

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

func (a *sigmoid) Backward(gradient u.Matrix) (u.Matrix, error) {

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
