package layers

import (
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type flatten struct {
  inShape t.Shape
  units int
}

func (f *flatten) CompileLayer(inShape t.Shape) (t.Shape, error) {
  f.inShape = inShape
  units := inShape.TotalSize()

  return []int{1, units}, nil 
}

func (f *flatten) Forward(input t.Tensor) (t.Tensor, error) {
  batches := input.Shape().Batches()

  err := input.Reshape([]int{batches, f.units})
  if err != nil {
    return nil, err
  }

  return input, nil
}

func (f *flatten) Backward(gradient t.Tensor) (t.Tensor, error) {
  
  err := gradient.Reshape(f.inShape)
  if err != nil {
    return nil, err
  }
  return gradient, nil
}

func (f *flatten) Type() string {
  return "flatten"
}

func (f *flatten) Weights() t.Tensor {
  return nil
}

func (f *flatten) Biases() t.Tensor {
  return nil
}

func (f *flatten) WeightsGradient() t.Tensor {
  return nil
}

func (f *flatten) BiasesGradient() t.Tensor {
  return nil
}

