package layers

import (
	"fmt"
	"math"

	u "github.com/cangeroe7/giraffe/internal/utils"
	a "github.com/cangeroe7/giraffe/pgk/activations"
)

type dense struct {
	units           int
	input           u.Matrix
	weights         u.Matrix
	biases          u.Matrix
	weightsGradient u.Matrix
	biasesGradient  u.Matrix
	activation      a.Activation
}

func Dense(units int, activation string) Layer {
	activationFunc := a.Activations[activation]()
	return &dense{units: units, activation: activationFunc}
}

func (d *dense) CompileLayer(inShape []int) ([]int, error) {
	var limit float64
	if d.activation.Type() == "relu" {
		// He initialization
		limit = math.Sqrt(2.0 / float64(inShape[1]))
		d.weights = u.RandMatrix(inShape[1], d.units, -limit, limit)
	} else {
		// Xavier initialization
		limit = math.Sqrt(6.0 / float64(inShape[1]+d.units))
		d.weights = u.RandMatrix(inShape[1], d.units, -limit, limit)
	}

	// Biases set to zero
	d.biases = u.ZerosMatrix(1, d.units)

	return d.biases.Shape(), nil
}

func (d *dense) Forward(input u.Matrix) (u.Matrix, error) {

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

func (d *dense) Backward(gradient u.Matrix) (u.Matrix, error) {
	// activation layer backward propagation code
	gradient, err := d.activation.Backward(gradient)
  if err != nil {
    return nil, err
  }

  d.weightsGradient, err = d.input.Transpose().MatMul(gradient)
  if err != nil {
    return nil, err
  }

  d.biasesGradient, err = gradient.SumAxis(0)
	if err != nil {
		fmt.Printf("err after activation backward: %v\n", err)
		return nil, err
	}

	outputGradient, err := gradient.MatMul(d.weights.Transpose())
	if err != nil {
		return nil, err
	}

	return outputGradient, nil
}

func (d *dense) WeightsGradient() u.Matrix {
  return d.weightsGradient
}

func (d *dense) BiasesGradient() u.Matrix {
  return d.biasesGradient
}


func (d *dense) Weights() u.Matrix {
  return d.weights
}

func (d *dense) Biases() u.Matrix {
  return d.biases
}

	// Adjust Weights
	// weights gradient = gradient * (input transposed)
	// adjusted weights = oldWeights - learningRate * weightGradient
	//inputT := d.input.Transpose()

	//weightsGradient, err := inputT.MatMul(deactiveGradient)
	//if err != nil {
	//	fmt.Printf("err weights gradient: %v\n", err)
	//	return nil, err
	//}
	// TODO: something with reps so the two things will multiply

	//d.weights.Add(weightsGradient.ScalarMultiply(-learningRate, false), false)

	// Adjust Biases
	// biases gradient = gradient
	// adjusted biases = oldbiases - learningRate * biases gradient
	//biasGradient, err := deactiveGradient.SumAxis(0)
	//if err != nil {
	//	fmt.Printf("err bias change: %v\n", err)
	//	return nil, err
	//}

	//d.biases.Add(biasGradient.ScalarMultiply(-learningRate, false), false)

	// Create Output Gradient
	// output gradient = gradient * (weights transposed)


