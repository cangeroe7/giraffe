package losses

import (
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Loss interface {
  CalcLoss(yTrue, yPred t.Tensor) (float64, error)
  Accuracy(yTrue, yPred t.Tensor) (float64, error)
  Gradient(yTrue, yPred t.Tensor) (t.Tensor, error)
}

func noZerosOnes(x float64) (float64, error) {
	epsilon := 1e-15
	if x < epsilon {
		return epsilon, nil
	}
	if x > 1-epsilon {
		return 1 - epsilon, nil
	}
	return x, nil
}
