package activations

import (
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Activation interface {
	Forward(input t.Tensor) (t.Tensor, error)
  Backward(gradient t.Tensor) (t.Tensor, error)
  Type() string
}

var Activations = map[string]func() Activation {
  "relu": Relu,
  "sigmoid": Sigmoid,
  "": nil,
  "none": nil,
  "linear": nil,
}
