package losses

import (
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)
var Losses = map[string]Loss{
  "mse": &meanSquareError{},
  "bce": &binaryCrossEntropy{},
}

type Loss interface {
  CalcLoss(yTrue, yPred t.Tensor) (float64, error)
  Accuracy(yTrue, yPred t.Tensor) (float64, error)
  Gradient(yTrue, yPred t.Tensor) (t.Tensor, error)
}
