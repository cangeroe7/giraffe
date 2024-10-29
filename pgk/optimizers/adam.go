package optimizers

import (
	"errors"

	u "github.com/cangeroe7/giraffe/internal/utils"
)

type Adam struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	firstM       map[string]u.Matrix // First moment
	secondV      map[string]u.Matrix // Second moment
	t            int                 // Time step
}

func (o *Adam) Apply(key string, param, gradient u.Matrix) error {
	o.t++
	//firstM := o.firstM
	//secondV := o.secondV

	return nil
}

func (a *Adam) Initialize() error {
	if a.LearningRate < 0.0 {
		return errors.New("Learning rate cannot be negative")
	}
	if a.LearningRate == 0.0 {
		a.LearningRate = 0.001
	}

	if a.Beta1 < 0.0 {
		return errors.New("Beta 1 cannot be negative")
	}
	if a.Beta1 == 0.0 {
		a.Beta1 = 0.9
	}

	if a.Beta2 < 0.0 {
		return errors.New("Beta 2 cannot be negative")
	}
	if a.Beta2 == 0.0 {
		a.Beta2 = 0.999
	}

	if a.Epsilon < 0.0 {
		return errors.New("Beta 1 cannot be negative")
	}
	if a.Epsilon == 0.0 {
		a.Epsilon = 1e-07
	}
  
  return nil
}
