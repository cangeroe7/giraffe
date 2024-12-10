package layers

import (

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

func NewPooling(poolType PoolingType, kernelSize, strides [2]int, mode PaddingMode, input t.Tensor) Layer {
  return &Pooling{
    PoolType: poolType,
    KernelSize: kernelSize,
    Strides: strides,
    Mode: mode,
    input: input,
  }
}
