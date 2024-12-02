package tensor

import (
	"errors"
	"strings"
)

type TensorIter interface {
  Next() (Tensor, bool)
  HasNext() bool
}

type tensorIter struct {
  t Tensor
  shape Shape
  step int
}

func (ti *tensorIter) Next() (Tensor, bool) {
  start := ti.shape.TotalSize() * ti.step
  end := ti.shape.TotalSize() * (ti.step + 1)

  if ti.t.Shape().TotalSize() < end {
    return nil, false
  }

  ti.step++
 
  return &tensor{TShape: ti.shape.Clone(), Data: (*ti.t.data())[start:end]}, true
}

func (ti *tensorIter) HasNext() bool {
  end := ti.shape.TotalSize() * (ti.step + 1)

  if ti.t.Shape().TotalSize() < end {
    return false
  }
  return true
}

func IterFromTensor(t Tensor, what string) (TensorIter, error) {
  if t.Dims() < 2 {
    return nil, errors.New("Cannot make iter over scalars")
  }

  var shape Shape
  switch strings.ToLower(what) {
  case "b", "batch", "batches":
    if t.Shape().IsMatrix() {
      shape = t.Shape().Clone()
      break
    }
    shape = t.Shape().Clone()[1:]

  case "r", "row", "rows":
    shape = []int{1, t.Shape()[len(t.Shape())-1]}

  case "c", "col", "column", "columns":
    shape = []int{1,1}

  // Defaults to matrices
  default:
    shape = t.Shape().Clone()[len(t.Shape())-2:]
  }

  return &tensorIter{t: t, shape: shape}, nil
}
