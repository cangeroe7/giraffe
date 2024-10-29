package losses

import (
	u "github.com/cangeroe7/giraffe/internal/utils"
)
var Losses = map[string]Loss{
  "mse": &meanSquareError{},
  "bce": &binaryCrossEntropy{},
}

type Loss interface {
  CalcLoss(yTrue, yPred u.Matrix) (float64, error)
  Accuracy(yTrue, yPred u.Matrix) (float64, error)
  Gradient(yTrue, yPred u.Matrix) (u.Matrix, error)
}
