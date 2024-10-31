package activations

import (
	u "github.com/cangeroe7/giraffe/internal/utils"
)

type Activation interface {
	Forward(input u.Matrix) (u.Matrix, error)
  Backward(gradient u.Matrix) (u.Matrix, error)
  Type() string
}

var Activations = map[string]func() Activation {
  "relu": Relu,
  "sigmoid": Sigmoid,
  "": nil,
  "none": nil,
  "linear": nil,
}
