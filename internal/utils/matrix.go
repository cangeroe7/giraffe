package utils

import (
	"errors"
	"fmt"
	"math/rand"
)

type Matrix interface {
	Append(x []float64) error
	MatMul(m_2 Matrix) (Matrix, error)
	Multiply(m_2 Matrix, inPlace bool) (Matrix, error)
	Divide(m_2 Matrix, inPlace bool) (Matrix, error)
	Add(m_2 Matrix, inPlace bool) (Matrix, error)
	Subtract(m_2 Matrix, inPlace bool) (Matrix, error)
	Map(fn func(float64) (float64, error), inPlace bool) (Matrix, error)
	MapOnto(fn func(float64, float64) (float64, error), m_2 Matrix, inPlace bool) (Matrix, error)
	ScalarMultiply(scalar float64, inPlace bool) Matrix
	ScalarDivide(scalar float64, inPlace bool) (Matrix, error)
	ScalarAdd(scalar float64, inPlace bool) Matrix
	RepAdd(m_2 Matrix, inPlace bool) (Matrix, error)
	Transpose() Matrix
	Sum() float64
	Avg() float64
	SumAxis(axis int) (Matrix, error)
	Min() float64
	Max() float64
	Shape() []int
	Size() int
	Tile(reps_rows, reps_cols int) (Matrix, error)
}

type matrix struct {
	mat [][]float64
}

func (m1 *matrix) Append(x []float64) error {
	if len(m1.mat[0]) != len(x) {
		return errors.New("Cannot append because of uneven lengths")
	}
	m1.mat = append(m1.mat, x)
	return nil
}

func (m1 *matrix) MatMul(m_2 Matrix) (Matrix, error) {
	m2, ok := m_2.(*matrix)
	if !ok {
		return nil, errors.New("invalid matrix type")
	}

	if len(m1.mat[0]) != len(m2.mat) {
		fmt.Printf("len(m1.mat[0]): %v\n", len(m1.mat[0]))
		fmt.Printf("len(m2.mat): %v\n", len(m2.mat))

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

func (m1 *matrix) Multiply(m_2 Matrix, inPlace bool) (Matrix, error) {
	m2, ok := m_2.(*matrix)
	if !ok {
		return nil, errors.New("invalid matrix type")
	}

	if len(m1.mat) != len(m2.mat) || len(m1.mat[0]) != len(m2.mat[0]) {
		return &matrix{}, errors.New("matrix dimensions are not compatible")
	}

	rows, cols := len(m1.mat), len(m1.mat[0])

	if inPlace {
		for r := range rows {
			for c := range cols {
				m1.mat[r][c] *= m2.mat[r][c]
			}
		}
		return m1, nil
	}

	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			mat[r][c] = m1.mat[r][c] * m2.mat[r][c]
		}
	}

	return &matrix{mat}, nil
}

func (m1 *matrix) Divide(m_2 Matrix, inPlace bool) (Matrix, error) {
	m2, ok := m_2.(*matrix)
	if !ok {
		return nil, errors.New("invalid matrix type")
	}

	if len(m1.mat) != len(m2.mat) || len(m1.mat[0]) != len(m2.mat[0]) {
		return &matrix{}, errors.New("matrix dimensions are not compatible")
	}

	rows, cols := len(m1.mat), len(m1.mat[0])

	if inPlace {
		for r := range rows {
			for c := range cols {
				if m2.mat[r][c] == 0.0 {
					return nil, errors.New("Division by 0")
				}
				m1.mat[r][c] /= m2.mat[r][c]
			}
		}
		return m1, nil
	}

	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			if m2.mat[r][c] == 0.0 {
				return nil, errors.New("Division by 0")
			}
			mat[r][c] = m1.mat[r][c] / m2.mat[r][c]
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

	if inPlace {
		for r := range rows {
			for c := range cols {
				m1.mat[r][c] += m2.mat[r][c]
			}
		}
		return m1, nil
	}

	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			mat[r][c] = m1.mat[r][c] + m2.mat[r][c]
		}
	}

	return &matrix{mat}, nil
}

func (m1 *matrix) RepAdd(m_2 Matrix, inPlace bool) (Matrix, error) {
	m2, ok := m_2.(*matrix)
	if !ok {
		return nil, errors.New("invalid matrix type")
	}

	rowsM1, colsM1 := len(m1.mat), len(m1.mat[0])
	rowsM2, colsM2 := len(m2.mat), len(m1.mat[0])

	if inPlace {
		for r := range rowsM1 {
			for c := range colsM1 {
				m1.mat[r][c] += m2.mat[r%rowsM2][c%colsM2]
			}
		}
		return m1, nil
	}
	mat := make([][]float64, rowsM1)
	for r := range rowsM1 {
		mat[r] = make([]float64, colsM1)
		for c := range colsM1 {
			mat[r][c] = m1.mat[r][c] + m2.mat[r%rowsM2][c%colsM2]
		}
	}
	return &matrix{mat}, nil
}

func (m1 *matrix) Subtract(m_2 Matrix, inPlace bool) (Matrix, error) {

	m2, ok := m_2.(*matrix)
	if !ok {
		return nil, errors.New("invalid matrix type")
	}

	if len(m1.mat) != len(m2.mat) || len(m1.mat[0]) != len(m2.mat[0]) {
		return &matrix{}, errors.New("matrix dimensions are not compatible")
	}

	rows, cols := len(m1.mat), len(m1.mat[0])

	if inPlace {
		for r := range rows {
			for c := range cols {
				m1.mat[r][c] -= m2.mat[r][c]
			}
		}
		return m1, nil
	}

	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			mat[r][c] = m1.mat[r][c] - m2.mat[r][c]
		}
	}

	return &matrix{mat}, nil
}

func(m *matrix) Map(fn func(float64) (float64, error), inPlace bool) (Matrix, error) {

	rows, cols := len(m.mat), len(m.mat[0])

	if inPlace {
		for r := range rows {
			for c := range cols {
				val, err := fn(m.mat[r][c])
				if err != nil {
					return nil, err
				}
				m.mat[r][c] = val
			}
		}
		return m, nil
	}

	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			val, err := fn(m.mat[r][c])

			if err != nil {
				return m, err
			}
			mat[r][c] = val
		}
	}

	return &matrix{mat}, nil
}

func (m1 *matrix) MapOnto(fn func(float64, float64) (float64, error), m_2 Matrix, inPlace bool) (Matrix, error) {

	m2, ok := m_2.(*matrix)
	if !ok {
		return nil, errors.New("invalid matrix type")
	}

	if len(m1.mat) != len(m2.mat) || len(m1.mat[0]) != len(m2.mat[0]) {
		return nil, errors.New("matrix dimensions are not compatible")
	}

	rows, cols := len(m1.mat), len(m1.mat[0])

	if inPlace {
		for r := range rows {
			for c := range cols {
				val, err := fn(m1.mat[r][c], m2.mat[r][c])
				if err != nil {
					return nil, err
				}
				m1.mat[r][c] = val
			}
		}
		return m1, nil
	}

	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			val, err := fn(m1.mat[r][c], m2.mat[r][c])

			if err != nil {
				return nil, err
			}
			mat[r][c] = val
		}
	}

	return &matrix{mat}, nil
}

func (m *matrix) Shape() []int {

	return []int{len(m.mat), len(m.mat[0])}
}

func (m *matrix) Sum() float64 {

	sum := 0.0
	for _, row := range m.mat {
		for _, val := range row {
			sum += val
		}
	}
	return sum
}
func (m *matrix) Avg() float64 {

	sum := 0.0
	for _, row := range m.mat {
		for _, val := range row {
			sum += val
		}
	}
	return sum / float64(m.Size())
}

func (m *matrix) Size() int {
	return len(m.mat) * len(m.mat[0])
}

func (m *matrix) SumAxis(axis int) (Matrix, error) {
	if axis != 0 && axis != 1 {
		return nil, errors.New("Not a valid axis, 0: X, 1: Y")
	}

	if len(m.mat) == 0 {
		return nil, errors.New("Empty matrix")
	}

	if axis == 0 {
		mat := make([][]float64, 1)
		mat[0] = make([]float64, len(m.mat[0]))
		for _, row := range m.mat {
			for c, val := range row {
				mat[0][c] += val
			}
		}
		return &matrix{mat}, nil
	}

	mat := make([][]float64, 1)
	mat[0] = make([]float64, len(m.mat))
	for r, row := range m.mat {
		for _, val := range row {
			mat[0][r] += val
		}
	}
	return &matrix{mat}, nil
}

func (m *matrix) Max() float64 {
	rows, cols := len(m.mat), len(m.mat[0])
	maxVal := m.mat[0][0]
	for r := range rows {
		for c := range cols {
			if m.mat[r][c] > maxVal {
				maxVal = m.mat[r][c]
			}
		}
	}
	return maxVal
}

func (m *matrix) Min() float64 {
	rows, cols := len(m.mat), len(m.mat[0])
	minVal := m.mat[0][0]
	for r := range rows {
		for c := range cols {
			if m.mat[r][c] < minVal {
				minVal = m.mat[r][c]
			}
		}
	}
	return minVal
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

func (m *matrix) Tile(reps_rows, reps_cols int) (Matrix, error) {
	if reps_rows == 0 || reps_cols == 0 {
		return nil, errors.New("Cannot have 0 repetitions")
	}

	rows, cols := len(m.mat), len(m.mat[0])

	tile := make([][]float64, rows*reps_rows)
	for r := range rows * reps_rows {
		tile[r] = make([]float64, cols*reps_cols)
		for c := range cols * reps_cols {
			tile[r][c] = m.mat[r%rows][c%cols]
		}
	}
	return &matrix{tile}, nil
}

func Identity(n int) (Matrix, error) {
	if n == 0 {
		return nil, errors.New("Cannot have identity of size 0")
	}

	identity := make([][]float64, n)
	for i := range n {
		identity[i] = make([]float64, n)
		for j := range n {
			if i == j {
				identity[i][j] = 1
			}
		}
	}

	return &matrix{identity}, nil
}

func (m *matrix) ScalarMultiply(scalar float64, inPlace bool) Matrix {
	rows, cols := len(m.mat), len(m.mat[0])
	if inPlace {
		for r := range len(m.mat) {
			for c := range len(m.mat[0]) {
				m.mat[r][c] *= scalar
			}
		}
		return m
	}
	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			mat[r][c] = m.mat[r][c] * scalar
		}
	}
	return &matrix{mat}
}

func (m *matrix) ScalarDivide(scalar float64, inPlace bool) (Matrix, error) {
	if scalar == 0 {
		return nil, errors.New("Cannot divide by zero")
	}
	rows, cols := len(m.mat), len(m.mat[0])
	if inPlace {
		for r := range len(m.mat) {
			for c := range len(m.mat[0]) {
				m.mat[r][c] /= scalar
			}
		}
		return m, nil
	}
	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			mat[r][c] = m.mat[r][c] / scalar
		}
	}
	return &matrix{mat}, nil
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

func FromMatrix(mat [][]float64) (Matrix, error) {
	rows := len(mat)

	if rows == 0 {
		return nil, errors.New("Empty matrix given")
	}

	cols := len(mat[0])

	if cols == 0 {
		return nil, errors.New("Empty matrix column given")
	}

	for r := range rows {
		if cols != len(mat[r]) {
			return nil, errors.New("Matrix columns lenghts do not match")
		}
	}

	return &matrix{mat}, nil
}

func RandMatrix(rows, cols int, min, max float64) Matrix {
	mat := make([][]float64, rows)
	for r := range rows {
		mat[r] = make([]float64, cols)
		for c := range cols {
			mat[r][c] = min + rand.Float64()*(max-min)
		}
	}

	return &matrix{mat}
}

func ZerosMatrix(rows, cols int) Matrix {

	mat := make([][]float64, rows)
	for i := range rows {
		mat[i] = make([]float64, cols)
	}

	return &matrix{mat}
}
