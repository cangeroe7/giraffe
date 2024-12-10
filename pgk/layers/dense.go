package layers

import (
	"errors"
	"fmt"
	"math"

	a "github.com/cangeroe7/giraffe/pgk/activations"
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Dense struct {
	Units      int
	Activation a.Activation

	input           t.Tensor
	weights         t.Tensor
	biases          t.Tensor
	weightsGradient t.Tensor
	biasesGradient  t.Tensor
}

func (d *Dense) Type() string {
	return "Dense"
}

func (d *Dense) Params() map[string]interface{} {
	return map[string]interface{}{
		"units":      d.Units,
		"activation": d.Activation.Type(),
	}
}

func (d *Dense) CompileLayer(inShape t.Shape) (t.Shape, error) {
	var limit float64
	if d.Activation.Type() == "relu" {
		// He initialization
		limit = math.Sqrt(2.0 / float64(inShape.Cols()))
		d.weights, _ = t.RandTensor([]int{inShape.Cols(), d.Units}, -limit, limit)
	} else {
		// Xavier initialization
		limit = math.Sqrt(6.0 / float64(inShape.Cols()+d.Units))
		d.weights, _ = t.RandTensor([]int{inShape.Cols(), d.Units}, -limit, limit)

	}

	// Biases set to zero
	d.biases = t.ZerosTensor([]int{1, d.Units})

	return d.biases.Shape(), nil
}

func (d *Dense) Forward(input t.Tensor) (t.Tensor, error) {
	if input == nil {
		return nil, errors.New("input cannot be nil")
	}

	// For when Dense comes in as (batches, 1, 1, cols)
	totalSize := input.Shape().TotalSize()
	rows := totalSize / input.Shape().Cols()
	input.Reshape([]int{1, 1, rows, input.Shape().Cols()})

	d.input = input
	Y, err := input.MatMul(d.weights)

	if err != nil {
		return nil, err
	}

	Y.RepAdd(d.biases, true)

	YActivated, err := d.Activation.Forward(Y)

	if err != nil {
		return nil, err
	}

	return YActivated, nil
}

func (d *Dense) Backward(gradient t.Tensor) (t.Tensor, error) {
	if gradient == nil {
		return nil, errors.New("gradient cannot be nil")
	}

	gradient, err := d.Activation.Backward(gradient)
	if err != nil {
		return nil, err
	}

	d.weightsGradient, err = d.input.Transpose(false).MatMul(gradient)
	if err != nil {
		return nil, err
	}

	d.biasesGradient, err = gradient.AxisSum(0)
	if err != nil {
		fmt.Printf("err after activation backward: %v\n", err)
		return nil, err
	}

	outputGradient, err := gradient.MatMul(d.weights.Transpose(false))
	if err != nil {
		return nil, err
	}

	return outputGradient, nil
}

func (d *Dense) WeightsGradient() t.Tensor {
	return d.weightsGradient
}

func (d *Dense) BiasesGradient() t.Tensor {
	return d.biasesGradient
}

func (d *Dense) Weights() t.Tensor {
	return d.weights
}

func (d *Dense) Biases() t.Tensor {
	return d.biases
}

func DenseFromParams(params map[string]interface{}, weights []float64, biases []float64) (Layer, error) {
  unitsFloat64, ok := params["units"].(float64)
  if !ok {
    return nil, errors.New("missing or invalid 'units' parameter")
  }

  units := int(unitsFloat64)

  activation, ok := params["activation"].(string)
  if !ok {
    return nil, errors.New("missing or invalid 'activation' parameter")
  }

  activationStruct := a.Activations[activation]()

  weightsShape := []int{len(weights)/units, units}
  biasesShape := []int{1, units}

  weightsTensor, err := t.TensorFrom(weightsShape, weights)
  if err != nil {
    return nil, err
  }


  biasesTensor, err := t.TensorFrom(biasesShape, biases)
  if err != nil {
    return nil, err
  }

  return &Dense{
    Units: units,
    Activation: activationStruct,
    weights: weightsTensor,
    biases: biasesTensor,
  }, nil
}
