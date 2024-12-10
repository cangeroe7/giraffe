package activations

import (
	"errors"
	"fmt"
	"math"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Softmax struct {
	output t.Tensor
}

func softmax() Activation {
  return &Softmax{}
}

func (a *Softmax) Type() string {
	return "softmax"
}

func (a *Softmax) Forward(input t.Tensor) (t.Tensor, error) {

	batchIter, err := t.IterFromTensor(input, "rows")
	if err != nil {
		return nil, err
	}
	for batch, ok := batchIter.Next(); ok; batch, ok = batchIter.Next() {

		maxVal := batch.Max()
		softmax := func(x float64) (float64, error) {
			return math.Exp(x - maxVal), nil
		}

		expValues, err := batch.Map(softmax, true)
		if err != nil {
			return nil, err
		}

		batchSum := expValues.Sum()
		_, err = expValues.ScalarDivide(batchSum, true)
		if err != nil {
			return nil, err
		}
	}

	a.output = input

	return a.output, nil
}

func (a *Softmax) Backward(gradient t.Tensor) (t.Tensor, error) {

	softmax := a.output

	outputGradient := t.ZerosTensor(gradient.Shape().Clone())

	softmaxIter, err := t.IterFromTensor(softmax, "rows")
	if err != nil {
		return nil, err
	}

	outGradientIter, err := t.IterFromTensor(outputGradient, "rows")
	if err != nil {
		return nil, err
	}

	gradientIter, err := t.IterFromTensor(gradient, "rows")
	if err != nil {
		return nil, err
	}

	for row, ok := softmaxIter.Next(); ok; row, ok = softmaxIter.Next() {
		//

		// softmax' = the idenitty - repeated rowSoftmax,
		// the result repeated multiplied by the transposed softmax row

		//
		outGradient, _ := outGradientIter.Next()
    gradient, _ := gradientIter.Next()

		size := row.Shape().TotalSize()

		rowT := row.Transpose(false)

		identity, _ := t.Identity(size)

		// No broadcast/repadd function for subtraction
		row.ScalarMultiply(-1.0, true)

		identity.RepAdd(row, true)

		identity.RepMultiply(rowT, true)

    gradient.Transpose(true)

    outputGradient, err := identity.MatMul(gradient)
    if err !=  nil {
      return nil, err
    }

    _, err = outGradient.Add(outputGradient.Transpose(true), true)
    if err != nil {
      return nil, err
    }
	}

  return outputGradient, nil
}

func (a *Softmax) wackBackward(gradient t.Tensor) (t.Tensor, error) {
	softmaxGradient, err := a.Forward(gradient)
	if err != nil {
		return nil, err
	}

	softmaxGradient.Reshape([]int{softmaxGradient.Shape().Cols(), 1})

	fmt.Printf("softmaxGradient.Shape(): %v\n", softmaxGradient.Shape())

	eye, err := t.Identity(softmaxGradient.Shape().Rows())
	fmt.Printf("eye.Shape(): %v\n", eye.Shape())
	eye.Print()

	tiledRowsAreSame, err := softmaxGradient.Tile(1, softmaxGradient.Shape().Rows())
	tiledColsAreSame, err := softmaxGradient.Transpose(false).Tile(1, softmaxGradient.Shape().Rows())
	if err != nil {
		return nil, err
	}
	fmt.Printf("tiledColsAreSame: %v\n", tiledColsAreSame)

	fmt.Printf("tiledRowsAreSame.Shape(): %v\n", tiledRowsAreSame.Shape())
	fmt.Println(tiledRowsAreSame.Slice(0, 50))

	fmt.Printf("tiledColsAreSame.Shape(): %v\n", tiledColsAreSame.Shape())
	fmt.Println(tiledColsAreSame.Slice(0, 50))

	return nil, errors.New("Implementing softmax")
}
