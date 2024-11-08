package layers

import (
	"fmt"
	"math"

	a "github.com/cangeroe7/giraffe/pgk/activations"
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type dense struct {
	units           int
	input           t.Tensor
	weights         t.Tensor
	biases          t.Tensor
	weightsGradient t.Tensor
	biasesGradient  t.Tensor
	activation      a.Activation
}

func Dense(units int, activation string) Layer {
	activationFunc := a.Activations[activation]()
	return &dense{units: units, activation: activationFunc}
}

func (d *dense) Type() string {
  return "dense"
}

func (d *dense) CompileLayer(inShape []int) ([]int, error) {
	var limit float64
	if d.activation.Type() == "relu" {
		// He initialization
		limit = math.Sqrt(2.0 / float64(inShape[1]))
		d.weights, _ = t.RandTensor([]int{inShape[1], d.units}, -limit, limit)
	} else {
		// Xavier initialization
		limit = math.Sqrt(6.0 / float64(inShape[1]+d.units))
		d.weights, _ = t.RandTensor([]int{inShape[1], d.units}, -limit, limit)
    
	}

	// Biases set to zero
	d.biases = t.ZerosTensor([]int{1, d.units})

	return d.biases.Shape(), nil
}

func (d *dense) Forward(input t.Tensor) (t.Tensor, error) {

	d.input = input
	Y, err := input.MatMul(d.weights)

	if err != nil {
		return nil, err
	}

	Y.RepAdd(d.biases, true)

	YActivated, err := d.activation.Forward(Y)

	if err != nil {
		return nil, err
	}

	return YActivated, nil
}

func (d *dense) Backward(gradient t.Tensor) (t.Tensor, error) {
	// activation layer backward propagation code

	gradient, err := d.activation.Backward(gradient)
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

func (d *dense) WeightsGradient() t.Tensor {
  return d.weightsGradient
}

func (d *dense) BiasesGradient() t.Tensor {
  return d.biasesGradient
}


func (d *dense) Weights() t.Tensor {
  return d.weights
}

func (d *dense) Biases() t.Tensor {
  return d.biases
}
