package layers

import (
	"errors"
	"math"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Layer interface {
	CompileLayer(inShape t.Shape) (t.Shape, error)
	Forward(input t.Tensor) (t.Tensor, error)
	Backward(gradient t.Tensor) (t.Tensor, error)
	Type() string

	Weights() t.Tensor
	Biases() t.Tensor
	WeightsGradient() t.Tensor
	BiasesGradient() t.Tensor
	Params() map[string]interface{}
}

type PaddingMode string

const (
	Valid PaddingMode = "valid"
	Full  PaddingMode = "full"
)

func ComputePadding(shape t.Shape, kernelSize, strides [2]int, mode PaddingMode) ([]int, error) {

	if shape == nil {
		return nil, errors.New("shape cannot be nil")
	}

	var padding []int
	if mode == Valid {
		padding = make([]int, 4)
		return padding, nil
	}

	inHeight := shape.Rows()

	outHeight := int(math.Ceil(float64(inHeight) / float64(strides[0])))

	padHeight := ((outHeight-1)*strides[0] + kernelSize[0] - inHeight) / 2

	var B, T, R, L int
	if padHeight > 0 {
		B = strides[0] / 2
		T = strides[0] - B
	} else {
		B = padHeight
		T = padHeight
	}

	inWidth := shape.Cols()

	outWidth := int(math.Ceil(float64(shape.Cols()) / float64(strides[1])))

	padWidth := ((outWidth-1)*strides[1] + kernelSize[1] - inWidth) / 2

	if padHeight > 0 {
		R = strides[1] / 2
		L = strides[1] - R
	} else {
		R = padWidth
		L = padWidth
	}

	return []int{T, R, B, L}, nil
}

func interfaceToIntArray(input []interface{}) ([]int, error) {
	result := make([]int, len(input))
	for i, v := range input {
		if val, ok := v.(float64); ok {
			result[i] = int(val)
		} else {
			return nil, errors.New("input contains non-integer value")
		}
	}
  return result, nil
}
func InterfaceToFloat64Array(input []interface{}) ([]float64, error) {

	result := make([]float64, len(input))
	for i, v := range input {
		if val, ok := v.(float64); ok {
			result[i] = val
		} else {
			return nil, errors.New("input contains non-float64 value")
		}
	}
  return result, nil
}
