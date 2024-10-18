package utils

import (
	"errors"
	"math/rand"
)

type Matrix interface {
	MatMul(m_2 Matrix) (Matrix, error)
	Add(m_2 Matrix, inPlace bool) (Matrix, error)
  ScalarMul(scalar float64, inPlace bool) Matrix
  ScalarAdd(scalar float64, inPlace bool) Matrix
	Transpose() Matrix
}

type matrix struct {
	mat [][]float64
}

func (m1 *matrix) MatMul(m_2 Matrix) (Matrix, error) {
	m2, ok := m_2.(*matrix)
	if !ok {
		return nil, errors.New("invalid matrix type")
	}

	if len(m1.mat[0]) != len(m2.mat) {
		return &matrix{}, errors.New("matrix dimensions are not compatible")
	}

	// define the dimensions of the return matrix
	rows, cols := len(m1.mat), len(m2.mat[0])

	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
	}
	for r := range rows {
		for c := range cols {
			for i := range len(m2.mat) {
				mat[r][c] += m1.mat[r][i] * m2.mat[i][c]
			}
		}
	}

	return &matrix{mat}, nil
}

func (m1 *matrix) Add(m_2 Matrix, inPlace bool) (Matrix, error) {

	m2, ok := m_2.(*matrix)
	if !ok {
		return nil, errors.New("invalid matrix type")
	}

	if len(m1.mat) != len(m2.mat) || len(m1.mat[0]) != len(m2.mat[0]) {
		return &matrix{}, errors.New("matrix dimensions are not compatible")
	}

	rows, cols := len(m1.mat), len(m1.mat[0])

	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			mat[r][c] = m1.mat[r][c] + m2.mat[r][c]
		}
	}

	return &matrix{mat}, nil
}

func (m *matrix) Transpose() Matrix {
	rows, cols := len(m.mat[0]), len(m.mat)

	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			mat[r][c] = m.mat[c][r]
		}
	}
	return &matrix{mat}
}

func (m *matrix) ScalarMul(scalar float64, inPlace bool) Matrix {
	rows, cols := len(m.mat), len(m.mat[0])
	if inPlace {
		for r := range len(m.mat) {
			for c := range len(m.mat[0]) {
				m.mat[r][c] *= scalar
			}
		}
		return m
	} else {
		mat := make([][]float64, rows)
		for r := range rows {
			mat[r] = make([]float64, cols)
			for c := range cols {
				mat[r][c] = m.mat[r][c] * scalar
			}
		}
		return &matrix{mat}
	}
}

func (m *matrix) ScalarAdd(scalar float64, inPlace bool) Matrix {
	rows, cols := len(m.mat), len(m.mat[0])

	if inPlace {
		for r := range len(m.mat) {
			for c := range len(m.mat[0]) {
				m.mat[r][c] += scalar
			}
		}
		return m
	} else {
		mat := make([][]float64, rows)
		for r := range rows {
			mat[r] = make([]float64, cols)
			for c := range cols {
				mat[r][c] = m.mat[r][c] + scalar
			}
		}
		return &matrix{mat}
	}
}

func MakeMat(values *[][]float64) (Matrix, error) {
	rows := len(*values)

	if rows < 1 {
		return &matrix{}, errors.New("Empty matrix given")
	}

	cols := len((*values)[0])

	if cols < 1 {
		return &matrix{}, errors.New("Empty matrix collum given")
	}

	for r := range rows {
		if cols != len((*values)[r]) {
			return &matrix{}, errors.New("Matrix collums not all the same length")
		}
	}

	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		copy(mat[r], (*values)[r])
	}

	return &matrix{}, nil
}

func MakeRandMat(rows, cols int, min, max float64) Matrix {
	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			mat[r][c] = min + rand.Float64()*(max-min)
		}
	}

	return &matrix{mat}
}

func MakeEmptyMat(rows, cols int) Matrix {

	mat := make([][]float64, rows)
	for i := range rows {
		mat[i] = make([]float64, cols)
	}

	return &matrix{mat}
}
