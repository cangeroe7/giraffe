package layers

import (
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Layer interface {
  Forward(input t.Tensor) (t.Tensor, error)
  Backward(gradient t.Tensor) (t.Tensor, error)
  CompileLayer(inShape []int) ([]int, error)
  Weights() t.Tensor
  Biases() t.Tensor
  WeightsGradient() t.Tensor
  BiasesGradient() t.Tensor
  Type() string
}


