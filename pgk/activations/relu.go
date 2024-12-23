package activations

import (
	"fmt"
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Relu struct {
	input t.Tensor
}

func relu() Activation {
  return &Relu{}
}

func (a *Relu) Type() string {
  return "relu"
}

func (a *Relu) Forward(input t.Tensor) (t.Tensor, error) {
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

func (a *Relu) Backward(gradient t.Tensor) (t.Tensor, error) {
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
