package losses

import (
	"errors"
	"math"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

// Binary Cross Entropy uses the function f(x) = yTrue * log(yPred) + (1 - yTrue) * log(1 - yPred)

type binaryCrossEntropy struct{}

func (l *binaryCrossEntropy) CalcLoss(yTrue, yPred t.Tensor) (float64, error) {
	trueShape, predShape := yTrue.Shape(), yPred.Shape()
	if len(trueShape) != len(predShape) {
		for i := range trueShape {
			if trueShape[i] != trueShape[i] {
				return 0.0, errors.New("Dimensions do not match")
			}
		}
	}

  BCE := func(vals ...float64) (float64, error) {
    if len(vals) != 2 {
      return 0.0, errors.New("More or less than 2 inputs")
    }
    x, y := vals[0], vals[1]
		return x*math.Log(y) + (1-x)*math.Log(1-y), nil
	}

	yPredClipped, _ := yPred.Map(noZerosOnes, false)

	losses, _ := yTrue.MapBatch(BCE, false, yPredClipped)

	loss := losses.Avg()

	return -loss, nil
}

func (l *binaryCrossEntropy) Accuracy(yTrue, yPred t.Tensor) (float64, error) {
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

func (l *binaryCrossEntropy) Gradient(yTrue, yPred t.Tensor) (t.Tensor, error) {
  if !yTrue.Shape().Eq(yPred.Shape()) {
    return nil, errors.New("Dimensions do not match")
  }

	primeBCE := func(vals ...float64) (float64, error) {
    if len(vals) != 2 {
      return 0.0, errors.New("More or less than 2 inputs")
    }

    x, y := vals[0], vals[1]
		if y == 0.0 {
			return 0.0, nil
		}
		return -x/y + (1-x)/(1-y), nil
	}

	yPredClipped, _ := yPred.Map(noZerosOnes, false)
	lossGradient, err := yTrue.MapBatch(primeBCE, false, yPredClipped)
	if err != nil {
		return nil, err
	}

	return lossGradient, nil
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
