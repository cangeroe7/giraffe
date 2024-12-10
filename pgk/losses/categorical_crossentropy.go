package losses

import (
	"math"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type CategoricalCrossentropy struct{}

func (l *CategoricalCrossentropy) CalcLoss(yTrue, yPred t.Tensor) (float64, error) {

  yPredClipped, err := yPred.Map(noZerosOnes, false)
  if err != nil {
    return 0.0, err
  }

  yPredLog, err := yPredClipped.Map(func (x float64) (float64, error) {
    return math.Log(x), nil
  }, true)
  if err != nil {
    return 0.0, err
  }

  lossTen, err := yTrue.Multiply(yPredLog, false)
  if err != nil {
    return 0.0, err
  }

  loss := -lossTen.Sum() / float64(yTrue.Shape().Rows())

	return loss, nil
}

func (l *CategoricalCrossentropy) Accuracy(yTrue, yPred t.Tensor) (float64, error) {

  yPredClasses, err := yPred.ArgMax(1)
  if err != nil {
    return 0.0, err
  }

  yTrueClasses, err := yTrue.ArgMax(1)
  if err != nil {
    return 0.0, err
  }

  correctCount := 0
  for i := range yTrueClasses {
    if yTrueClasses[i] == yPredClasses[i] {
      correctCount++
    }
  }

  accuracy := float64(correctCount) / float64(len(yTrueClasses))

	return accuracy, nil
}

func (l *CategoricalCrossentropy) Gradient(yTrue, yPred t.Tensor) (t.Tensor, error) {

	yPredClipped, _ := yPred.Map(noZerosOnes, false)

  gradient := yTrue.ScalarMultiply(-1.0, false)

  gradient, _ = gradient.Divide(yPredClipped, true)

	return gradient, nil
}
