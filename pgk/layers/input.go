package layers

import (
	"errors"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Input struct {
	Shape t.Shape
}

func (i *Input) Type() string {
	return "Input"
}

func (i *Input) Params() map[string]interface{} {
  return map[string]interface{}{
    "shape": i.Shape,
  }
}

func (i *Input) Forward(input t.Tensor) (t.Tensor, error) {

	inShape := input.Shape()

	if inShape.Channels() != i.Shape.Channels() {
		return nil, errors.New("Channels dimsize does not match")
	}

	if inShape.Rows() != i.Shape.Rows() {
		return nil, errors.New("Row dimsize does not match")
	}

	if inShape.Cols() != i.Shape.Cols() {
		return nil, errors.New("Column dimsize does not match")
	}

	return input, nil
}

func (i *Input) Backward(gradient t.Tensor) (t.Tensor, error) {
	return gradient, nil
}

func (i *Input) CompileLayer(inShape t.Shape) (t.Shape, error) {
	return inShape, nil
}

func (i *Input) Weights() t.Tensor {
	return nil
}

func (i *Input) Biases() t.Tensor {
	return nil
}

func (i *Input) WeightsGradient() t.Tensor {
	return nil
}

func (i *Input) BiasesGradient() t.Tensor {
	return nil
}

func InputFromParams(params map[string]interface{}) (Layer, error){

  interfaceShape, ok := params["shape"].([]interface{})
  if !ok {
    return nil, errors.New("missing or invalid 'shape' parameter")
  }

  shape, err := interfaceToIntArray(interfaceShape)
  if err != nil {
    return nil, err
  }

  return &Input{
    Shape: shape,
  }, nil
}
