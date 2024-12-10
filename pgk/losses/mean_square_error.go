package losses

import (
	"errors"
	"math"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type MeanSquareError struct{}

func (l *MeanSquareError) CalcLoss(yTrue, yPred t.Tensor) (float64, error) {
	trueShape, predShape := yTrue.Shape(), yPred.Shape()
	if len(trueShape) != len(predShape) {
		for i := range trueShape {
			if trueShape[i] != trueShape[i] {
				return 0.0, errors.New("Dimensions do not match")
			}
		}
	}
	diffs, _ := yTrue.Subtract(yPred, false)
	squaredDiffs, _ := diffs.Multiply(diffs, true)

  avg := squaredDiffs.Avg()

  return avg, nil
}

func (l *MeanSquareError) Accuracy(yTrue, yPred t.Tensor) (float64, error) {
	round := func(x float64) (float64, error) {
		return math.Round(x), nil
	}

	predicted, _ := yPred.Map(round, false)
	predicted.Add(yTrue, true)

	correct := func(x float64) (float64, error) {
		if x == 2.0 || x == 0.0 {
			return 1.0, nil
		}
		return 0.0, nil
	}

	predicted.Map(correct, true)
	sum := predicted.Sum()
	return sum / float64(predicted.Size()), nil
}

func (l *MeanSquareError) Gradient(yTrue, yPred t.Tensor) (t.Tensor, error) {
	trueShape, predShape := yTrue.Shape(), yPred.Shape()
	if len(trueShape) != len(predShape) {
		for i := range trueShape {
			if trueShape[i] != trueShape[i] {
				return nil, errors.New("Dimensions do not match")
			}
		}
	}

	size := yTrue.Size()
	diffs, err := yPred.Subtract(yTrue, false)
	if err != nil {
		return nil, err
	}
	gradient := diffs.ScalarMultiply(2.0/float64(size), true)

	return gradient, nil
}
