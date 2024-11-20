package layers

import (
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Layer interface {
  CompileLayer(inShape t.Shape) (t.Shape, error)
  Forward(input t.Tensor) (t.Tensor, error)
  Backward(gradient t.Tensor) (t.Tensor, error)
  Type() string
  Weights() t.Tensor
  Biases() t.Tensor
  WeightsGradient() t.Tensor
  BiasesGradient() t.Tensor
}


