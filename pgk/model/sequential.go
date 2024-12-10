package model

import (
	"errors"
	"fmt"
	"math"
	"time"

	la "github.com/cangeroe7/giraffe/pgk/layers"
	lo "github.com/cangeroe7/giraffe/pgk/losses"
	o "github.com/cangeroe7/giraffe/pgk/optimizers"
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Model interface {
	Add(layer la.Layer)
	Compile(inputShape []int, optimizer o.Optimizer, loss string, metrics []string, compileLayers bool) error
	Fit(xTrain, yTrain t.Tensor, batchSize int, epochs int) error
	History(metrics ...string) map[string]([]float64)
	Evaluate(input [][]float64) (t.Tensor, error)
	SaveModel(path string) error
}

type sequential struct {
	layers    []la.Layer
	optimizer o.Optimizer
	loss      lo.Loss
	batchSize int
	epochs    int
	history   map[string]([]float64)
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

func (s *sequential) Compile(inputShape []int, loss lo.Loss, optimizer o.Optimizer, metrics []string, compileLayers bool) error {
	// TODO: Check if all layers can be or are properly connected to each other
	if compileLayers {
		outputShape := inputShape
		for _, layer := range s.layers {
			var err error
			outputShape, err = layer.CompileLayer(outputShape)
			if err != nil {
				return err
			}
		}
	}

	// Set loss
	if loss == nil {
		return errors.New("loss has to be assigned")
	}

	s.loss = loss

	// Set optimzers; stochatic gradient descent as default
	if optimizer == nil {
		optimizer = &o.Adam{}
	}
	s.optimizer = optimizer
	s.optimizer.Initialize()

	return nil
}

func (s *sequential) Fit(xTrain, yTrain t.Tensor, batchSize, epochs int, normalize bool) error {

	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("\nEpoch %v  \n", epoch+1)

		start := time.Now()
		totalLoss := 0.0
		totalAccuracy := 0.0

		// Shuffle data for each epoch to prevent overfitting to fixed batches
		t.Shuffle(xTrain, yTrain)

		numBatches := (xTrain.Shape().Batches() + batchSize - 1) / batchSize

    batchStep := numBatches / 50
    diff := numBatches % 50 

		for batch := 0; batch < numBatches; batch++ {
			if (batch - diff)%(batchStep+1) == 0 {
				fmt.Printf("#")
			}

			// Get current batch of data
			start := batch * batchSize
			end := min(start+batchSize, xTrain.Shape().Batches()-1)

			XBatch, err := xTrain.BatchSlice(start, end) //t.TensorFromMatrix(&subXTrain)
			if err != nil {
				return err
			}

			YBatch, err := yTrain.BatchSlice(start, end) //t.TensorFromMatrix(&subXTrain)
			if err != nil {
				return err
			}
			YBatch.Reshape([]int{YBatch.Shape().Batches(), YBatch.Shape().Cols()})

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
		s.history["loss"] = append(s.history["loss"], math.Round((totalLoss*10000)/float64(numBatches))/10000)
		s.history["accuracy"] = append(s.history["accuracy"], math.Round((totalAccuracy*10000)/float64(numBatches))/10000)

		duration := time.Since(start)
		fmt.Printf(" Time: %v\n", duration)
    fmt.Printf("Metrics: [ loss: %v, accuracy: %v ]\n", totalLoss/float64(numBatches), totalAccuracy/float64(numBatches))


	}

	return nil
}

func (s *sequential) History(metrics ...string) map[string]([]float64) {
	fmt.Printf("loss: %v\n", s.history["loss"])
	fmt.Printf("accuracy: %v\n", s.history["accuracy"])
	return s.history
}

func (s *sequential) Evaluate(input t.Tensor) (t.Tensor, error) {
	// Normalize data if normalization is used

	output := input
	var err error
	for _, layer := range s.layers {
		output, err = layer.Forward(output)
		if err != nil {
			return nil, err
		}
	}
	round := func(x float64) (float64, error) {
		return math.Round(x*10000) / 10000, nil
	}

	_, _ = output.Map(round, true)

	return output, nil
}
