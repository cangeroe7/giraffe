package layers

import (
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Flatten struct {
	inShape t.Shape
}

func (f *Flatten) CompileLayer(inShape t.Shape) (t.Shape, error) {
	return []int{1, inShape.TotalSize()}, nil
}

func (f *Flatten) Forward(input t.Tensor) (t.Tensor, error) {
	batches := input.Shape().Batches()
	f.inShape = input.Shape().Clone()
	units := f.inShape.TotalSize() / batches

	err := input.Reshape([]int{batches, units})
	if err != nil {
		return nil, err
	}

	return input, nil
}

func (f *Flatten) Backward(gradient t.Tensor) (t.Tensor, error) {

	err := gradient.Reshape(f.inShape)
	if err != nil {
		return nil, err
	}
	return gradient, nil
}

func (f *Flatten) Type() string {
	return "Flatten"
}

func (f *Flatten) Params() map[string]interface{} {
	return map[string]interface{}{}
}

func (f *Flatten) Weights() t.Tensor {
	return nil
}

func (f *Flatten) Biases() t.Tensor {
	return nil
}

func (f *Flatten) WeightsGradient() t.Tensor {
	return nil
}

func (f *Flatten) BiasesGradient() t.Tensor {
	return nil
}

func FlattenFromParams() (Layer, error) {
  return &Flatten{}, nil
}
