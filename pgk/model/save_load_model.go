package model

import (
	"encoding/json"
	"errors"
	"os"

	l "github.com/cangeroe7/giraffe/pgk/layers"
)

type serializableLayer struct {
	Type    string                 `json:"type"`
	Params  map[string]interface{} `json:"params"`
	Weights []float64              `json:"weights,omitempty"`
	Biases  []float64              `json:"biases,omitempty"`
}

type serializableModel struct {
	Layers  []serializableLayer  `json:"layers"`
	History map[string][]float64 `json:"history"`
}

func (s *sequential) SaveModel(path string) error {

	serializedModel := serializableModel{
		History: s.history,
	}

	for _, layer := range s.layers {
		layerInfo := serializableLayer{
			Type:   layer.Type(),
			Params: layer.Params(),
		}

		if weights := layer.Weights(); weights != nil {
			layerInfo.Weights = weights.DataCopy()
		}

		if biases := layer.Biases(); biases != nil {
			layerInfo.Biases = biases.DataCopy()
		}
		serializedModel.Layers = append(serializedModel.Layers, layerInfo)
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	if err := encoder.Encode(serializedModel); err != nil {
		return err
	}

	return nil
}

// LoadModel loads a model's structure and parameters from a file
func LoadModel(path string) (*sequential, error) {
	// Open the JSON file
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Decode the JSON file into SerializableModel
	var serializedModel serializableModel
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&serializedModel); err != nil {
		return nil, err
	}

	// Reconstruct the sequential model
	model := Sequential()
	model.history = serializedModel.History

	// Recreate each layer using their Load function
	for _, layerInfo := range serializedModel.Layers {

		// Get the layer's Load function based on its type
		layer, err := loadLayer(layerInfo.Type, layerInfo.Params, layerInfo.Weights, layerInfo.Biases)
		if err != nil {
			return nil, err
		}

		// Add the reconstructed layer to the model
		model.Add(layer)
	}

	return model, nil
}

func loadLayer(layerType string, params map[string]interface{}, weights []float64, biases []float64) (l.Layer, error) {
	switch layerType {
	case "Dense":
		return l.DenseFromParams(params, weights, biases)
	case "Conv2D":
		return l.Conv2DFromParams(params, weights, biases)
	case "Pooling":
		return l.PoolingFromParams(params)
	case "Flatten":
		return l.FlattenFromParams()
	case "Input":
		return l.InputFromParams(params)
	default:
		return nil, errors.New("LoadLayer() Error: Invalid layer type")
	}
}
