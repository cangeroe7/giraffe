package optimizers

import (
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Optimizer interface {
  Initialize() error
	Apply(key string, param, gradient t.Tensor) error

}
