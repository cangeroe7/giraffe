package layers

import (
	u "github.com/cangeroe7/giraffe/internal/utils"
)

type Dense struct {
	units      int
	input      u.Matrix
	weights    u.Matrix
	biases     u.Matrix
	activation func(float64) float64
}

func (d *Dense) forward(input u.Matrix) (u.Matrix, error) {
	output, err := input.MatMul(d.weights)

	if err != nil {
		return output, err
	}

  output.Add(d.biases, true)

	// TODO: make it go through the activation layer
	return output, nil
}

func (d *Dense) backward(gradient u.Matrix, learningRate float64) (u.Matrix, error) {
	// TODO: do the activation layer backwards propagation first
	// activation layer backward propagation code

	// Adjust Weights
	// weights gradient = gradient * (input transposed)
	// adjusted weights = oldWeights - learningRate * weightGradient
	weightsGradient, err := gradient.MatMul(d.input.Transpose())

  if err != nil {
    return gradient, err
  }

  d.weights.Add(weightsGradient.ScalarMul(-learningRate, true), true)

  // Adjust Biases 
  // biases gradient = gradient
  // adjusted biases = oldbiases - learningRate * biases gradient
  d.biases.Add(gradient.ScalarMul(-learningRate, false), true)

  // Create Output Gradient 
  // output gradient = (weights transposed) * gradient
  outGradient, err := d.weights.Transpose().MatMul(gradient)

  if err != nil {
    return gradient, err
  }

  return outGradient, nil
}
