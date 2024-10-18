package activations

import (
	u "github.com/cangeroe7/giraffe/internal/utils"
)

type Activation interface {
  Forward(input u.Matrix)
  Backward(gradient u.Matrix)
}

