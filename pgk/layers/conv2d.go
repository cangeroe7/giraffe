package layers

import (
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type conv2D struct {
  shape t.Shape

  weights t.Tensor
  biases t.Tensor
  weightsGradient t.Tensor
  biasesGradient t.Tensor
}

func Conv2D(shape []int) Layer {
  return &conv2D{shape: shape}
}

func (c *conv2D) Type() string {
  return "Conv2D"
}

func (c *conv2D) Initialize(input t.Tensor) (t.Tensor, error) {

  return nil, nil
}


func (c *conv2D) Forward(input t.Tensor) (t.Tensor, error) {

  return nil, nil
}

func (c *conv2D) Backward(gradient t.Tensor) (t.Tensor, error) {

  return nil, nil
}

func (c *conv2D) CompileLayer(inShape []int) ([]int, error) {
  return nil, nil
}

func (c *conv2D) Weights() t.Tensor {
  return c.weights
}

func (c *conv2D) Biases() t.Tensor {
  return c.biases
}

func (c *conv2D) WeightsGradient() t.Tensor {
  return c.weightsGradient
}

func (c *conv2D) BiasesGradient() t.Tensor {
  return c.biasesGradient
}
