package optimizers

import (
	"errors"

	u "github.com/cangeroe7/giraffe/internal/utils"
)
type SGD struct {
  LearningRate float64
}

func (o *SGD) Initialize() error {
  if o.LearningRate < 0.0 {
    return errors.New("Learning rate has to be positive")
  } else if o.LearningRate == 0.0 {
    o.LearningRate = 0.01
  }
  
  return nil
}

func (o *SGD) Apply(key string, param, gradient u.Matrix) error {
  if param == nil || gradient == nil {
    return nil
  }

  scaledGradient := gradient.ScalarMultiply(o.LearningRate, false)

  _, err := param.Subtract(scaledGradient, true)
  if err != nil {
    return err
  }

  return nil
}
