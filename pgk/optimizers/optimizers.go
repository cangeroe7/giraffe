package optimizers

import (
	u "github.com/cangeroe7/giraffe/internal/utils"
)

type Optimizer interface {
  Initialize() error
	Apply(key string, param, gradient u.Matrix) error

}
