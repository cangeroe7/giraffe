package model

import (
	"fmt"
	"math"

	u "github.com/cangeroe7/giraffe/internal/utils"
	la "github.com/cangeroe7/giraffe/pgk/layers"
	lo "github.com/cangeroe7/giraffe/pgk/losses"
	o "github.com/cangeroe7/giraffe/pgk/optimizers"
)

type Model interface {
	Add(layer la.Layer)
	Compile(inputShape []int, optimizer o.Optimizer, loss string, metrics []string) error
	Fit(xTrain, yTrain [][]float64, batchSize int, epochs int) error
	History(metrics ...string) []float64
	Evaluate(input [][]float64) (u.Matrix, error)
}

type sequential struct {
	layers     []la.Layer
	optimizer  o.Optimizer
	loss       lo.Loss
	batchSize  int
	epochs     int
	history    map[string]([]float64)
	normalizer u.Normalizer
}

func Sequential(layers ...la.Layer) *sequential {
	var metrics = map[string]([]float64){"loss": []float64{}, "accuracy": []float64{}}
	model := sequential{history: metrics}
	for _, layer := range layers {
		model.Add(layer)
	}
	return &model
}

func (s *sequential) Add(layer la.Layer) {
	s.layers = append(s.layers, layer)
}

func (s *sequential) Compile(inputShape []int, loss string, optimizer o.Optimizer, metrics []string) error {
	// TODO: Check if all layers can be or are properly connected to each other
	outputShape := inputShape
	for _, layer := range s.layers {
		var err error
		outputShape, err = layer.CompileLayer(outputShape)
		if err != nil {
			return err
		}
	}

	// TODO: Check if the output layer is compatible with the loss function
	s.loss = lo.Losses[loss]
	// TODO: Initialize the metrics that we want to keep track off
	// TODO: Set up the optimizer for the model

	// Set optimzers; stochatic gradient descent as default
	if optimizer == nil {
		optimizer = &o.SGD{}
	}
	s.optimizer = optimizer
	s.optimizer.Initialize()

	return nil
}

func (s *sequential) Fit(xTrain, yTrain [][]float64, batchSize, epochs int, normalize bool) error {
	if normalize {
		s.normalizer.SetFeatures(&xTrain)
		s.normalizer.NormalizeData(&xTrain)
	}

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle data for each epoch to prevent overfitting to fixed batches
		totalLoss := 0.0
		totalAccuracy := 0.0
		u.Shuffle(xTrain, yTrain)

		numBatches := (len(xTrain) + batchSize - 1) / batchSize

		for batch := 0; batch < numBatches; batch++ {
			// Get current batch of data
			start := batch * batchSize
			end := min(start+batchSize, len(xTrain))

			XBatch, err := u.FromMatrix(xTrain[start:end])
			if err != nil {
				return err
			}
			YBatch, err := u.FromMatrix(yTrain[start:end])
			if err != nil {
				return err
			}

			// Forward pass
			output := XBatch
			for _, layer := range s.layers {
				output, err = layer.Forward(output)
				if err != nil {
					return err
				}
			}

			// Calculate loss and loss gradient
			loss, err := s.loss.CalcLoss(YBatch, output)
			if err != nil {
				return err
			}
			totalLoss += loss

			accuracy, err := s.loss.Accuracy(YBatch, output)
			if err != nil {
				return err
			}
			totalAccuracy += accuracy

			lossGradient, err := s.loss.Gradient(YBatch, output)
			if err != nil {
				return err
			}

			// Backward pass
			grad := lossGradient
			for i := len(s.layers) - 1; i >= 0; i-- {
				grad, err = s.layers[i].Backward(grad)
				if err != nil {
					return err
				}
			}

			// Update weights and biases for each layer
			for i, layer := range s.layers {
				s.optimizer.Apply(fmt.Sprintf("layer%d_weights", i+1), layer.Weights(), layer.WeightsGradient())
				s.optimizer.Apply(fmt.Sprintf("layer%d_biases", i+1), layer.Biases(), layer.BiasesGradient())
			}
		}

		// Append the loss and accuracy metrics
		s.history["loss"] = append(s.history["loss"], math.Round((totalLoss * 10000)/float64(numBatches))/10000)
		s.history["accuracy"] = append(s.history["accuracy"], math.Round((totalAccuracy * 10000)/float64(numBatches))/10000)
	}

	return nil
}

func (s *sequential) History(metrics ...string) map[string][]float64 {
	fmt.Printf("loss: %v\n", s.history["loss"])
	fmt.Printf("accuracy: %v\n", s.history["accuracy"])
	return s.history
}

func (s *sequential) Evaluate(data *[][]float64) (u.Matrix, error) {
	// Normalize data if normalization is used
	if s.normalizer.Activated {
		s.normalizer.NormalizeData(data)
	}

	input, err := u.FromMatrix(*data)
	if err != nil {
		return nil, err
	}

	output := input
	for _, layer := range s.layers {
		output, err = layer.Forward(output)
		if err != nil {
			return nil, err
		}
	}
  round := func(x float64) (float64, error) {
    return math.Round(x * 10000) / 10000, nil
  }

  _, _= output.Map(round, true)

	return output, nil
}
