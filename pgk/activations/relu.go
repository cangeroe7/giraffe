package activations

import (
	"fmt"
	u "github.com/cangeroe7/giraffe/internal/utils"
)

type relu struct {
	input u.Matrix
}

func Relu() Activation { return &relu{} }

func (a *relu) Type() string {
  return "relu"
}

func (a *relu) Forward(input u.Matrix) (u.Matrix, error) {
	a.input = input
	relu := func(x float64) (float64, error) {
		if x > 0 {
			return x, nil
		} else {
			return 0.0, nil
		}
	}

	output, err := input.Map(relu, false)

	if err != nil {
		return nil, err
	}

	return output, nil
}

func (a *relu) Backward(gradient u.Matrix) (u.Matrix, error) {
	reluPrime := func(x float64) (float64, error) {
		if x > 0 {
			return 1.0, nil
		} else {
			return 0.0, nil
		}
	}
	deactivated, err := a.input.Map(reluPrime, false)
	if err != nil {
		fmt.Printf("err after map in relu: %v\n", err)
		return nil, err
	}

	outGradient, err := gradient.Multiply(deactivated, false)
	if err != nil {
		fmt.Printf("err in creating output gradient for relu: %v\n", err)
		return nil, err
	}

	return outGradient, nil
}
