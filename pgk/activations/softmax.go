package activations

import (
	u "github.com/cangeroe7/giraffe/internal/utils"
	"math"
)

func Softmax() Activation { return &softmax{} }

type softmax struct {
	output u.Matrix
}

func (a *softmax) Type() string {
  return "softmax"
}

func (a *softmax) Forward(input u.Matrix) (u.Matrix, error) {
	maxVal := input.Max()
	softmax := func(x float64) (float64, error) {
		return math.Exp(x - maxVal), nil
	}

	expValues, err := input.Map(softmax, false)
	if err != nil {
		return nil, err
	}

	matSum := input.Sum()
	output, err := expValues.ScalarDivide(matSum, true)
	if err != nil {
		return nil, err
	}
	a.output = output

	return output, nil
}

func (a *softmax) Backward(gradient u.Matrix) (u.Matrix, error) {
	shape := a.output.Shape()
	size := shape[0] * shape[1]

	M, _ := a.output.Tile(1, size)
	MTransposed := M.Transpose()
	identity, _ := u.Identity(size)
	IminusM, _ := identity.Subtract(MTransposed, true)
	tmpXIminusM, _ := M.Multiply(IminusM, true)
	outputGradient, _ := tmpXIminusM.MatMul(gradient)

	return outputGradient, nil
}

