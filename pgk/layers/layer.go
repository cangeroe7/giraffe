package layers

import (
	u "github.com/cangeroe7/giraffe/internal/utils"
)

type Layer interface {
  Forward(input u.Matrix) (u.Matrix, error)
  Backward(gradient u.Matrix) (u.Matrix, error)
  CompileLayer(inShape []int) ([]int, error)
  Weights() u.Matrix
  Biases() u.Matrix
  WeightsGradient() u.Matrix
  BiasesGradient() u.Matrix
  Type() string
}


