package tensor

import (
	"errors"
)

type Shape []int

func (s Shape) IsMatrix() bool {
  return len(s) == 2
}

func (s Shape) CalcStrides() []int {
	if s.IsScalar() {
		return nil
	}
  acc := 1
  strides := make([]int, len(s))
  for i := len(s)-1; i >= 0; i-- {
    strides[i] = acc
    d := s[i]
    if d < 0 {
      panic("negative dimension size doesn't work")
    }
    acc *= d
  }

  return strides
}

func (s Shape) Transpose() Shape {
  s[0], s[1] = s[1], s[0]
  return s
}

func (s Shape) Clone() Shape {
  shape := make([]int, len(s))
  copy(shape, s)
  return shape
}

func (s Shape) DimSize(dim int) (size int, err error) {
	if (s.IsScalar() && dim != 0) || (!s.IsScalar() && dim >= len(s)) {
		err = errors.New("Not a valid dimension size option")
		return
	}

	switch {
	case s.IsScalar():
		return 0, nil
	default:
		return s[dim], nil
	}
}

func (s Shape) Dims() int {
	return len(s)
}

func (s Shape) Eq(other Shape) bool {
	if s.IsScalar() && other.IsScalar() {
		return true
	}

	if len(s) != len(other) {
		return false
	}

	for i := range len(s) {
		if s[i] != other[i] {
			return false
		}
	}

	return true
}

func (s Shape) TotalSize() int {
	if s.IsScalar() {
		return 0
	}

	totalSize := 1
	for _, dim := range s {
		totalSize *= dim
	}
	return totalSize
}

func (s Shape) IsScalar() bool {
	return len(s) == 0
}