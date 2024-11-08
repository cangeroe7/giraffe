package tensor

import (
	"errors"
	"math"
	"math/rand"
	"sync"
)

type Tensor interface {
	// Info about the ndarray
	Shape() Shape
	Strides() []int
	Dims() int
	Size() int

	// Data Access
	data() *[]float64

	// Stuff Like Transpose
	Transpose(inPlace bool) Tensor

	Map(fn func(float64) (float64, error), inPlace bool) (Tensor, error)
	MapBatch(fn func(...float64) (float64, error), inPlace bool, others ...Tensor) (Tensor, error)

	// Operations
	RepAdd(other Tensor, inPlace bool) (Tensor, error)
	Add(other Tensor, inPlace bool) (Tensor, error)
	Subtract(other Tensor, inPlace bool) (Tensor, error)
	Multiply(other Tensor, inPlace bool) (Tensor, error)
	Divide(other Tensor, inPlace bool) (Tensor, error)

	ScalarAdd(x float64, inPlace bool) Tensor
	ScalarSubtract(x float64, inPlace bool) Tensor
	ScalarMultiply(x float64, inPlace bool) Tensor
	ScalarDivide(x float64, inPlace bool) (Tensor, error)

	Sum() float64
	Avg() float64
	Min() float64
	Max() float64

	Tile(reps []int) (Tensor, error)
	AxisSum(axis int) (Tensor, error)

	// Matrix Multiplication Functions
	MatMul(other Tensor) (Tensor, error)
	BatchMatMul(other Tensor) (Tensor, error)
}

func ZerosTensor(shape Shape) Tensor {
	return &tensor{TShape: shape.Clone(), Data: make([]float64, shape.TotalSize())}
}

func RandTensor(shape Shape, minVal, maxVal float64) (Tensor, error) {
	if minVal > maxVal {
		return nil, errors.New("minVal bigger than maxVal")
	}

	resTen := tensor{TShape: shape.Clone(), Data: make([]float64, shape.TotalSize())}
	for i := range resTen.Data {
		resTen.Data[i] = minVal + rand.Float64()*(maxVal-minVal)
	}

	return &resTen, nil
}

func TensorFrom(shape Shape, input []float64) (Tensor, error) {
	if shape.TotalSize() != len(input) {
		return nil, errors.New("Shape doesn't match size of input")
	}

	return &tensor{TShape: shape.Clone(), Data: input}, nil
}

func TensorFromMatrix(input *[][]float64) (Tensor, error) {
	rows, cols := len(*input), len((*input)[0])

	if rows <= 0 || cols <= 0 {
		return nil, errors.New("cannot have an empty matrix")
	}

	resTen := tensor{TShape: []int{rows, cols}, Data: make([]float64, rows*cols)}

	for i := range rows {
		for j := range cols {
			resTen.Data[i*cols+j] = (*input)[i][j]
		}
	}

	return &resTen, nil
}

func Identity(n int) (Tensor, error) {
	if n <= 0 {
		return nil, errors.New("Cannot have identity less than or equal to zero")
	}

	resTen := tensor{TShape: []int{n, n}, Data: make([]float64, n*n)}

	for i := 0; i < n*n; i += n + 1 {
		resTen.Data[i] = 1.0
	}

	return &resTen, nil
}

type tensor struct {
	TShape Shape
	Data   []float64
}

func (t *tensor) Transpose(inPlace bool) Tensor {
	if !t.Shape().IsMatrix() {
		return nil
	}

	transposed := make([]float64, len(t.Data))

	rows := t.TShape[0]
	cols := t.TShape[1]

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j*rows+i] = t.Data[i*cols+j]
		}
	}

	if inPlace {
		t.Data = transposed
		t.TShape.Transpose()
		return t
	}

	return &tensor{TShape: t.Shape().Clone().Transpose(), Data: transposed}
}

func (t *tensor) RepAdd(other Tensor, inPlace bool) (Tensor, error) {

	if other.Dims() == 0 {
		return nil, errors.New("other tensor has no dimension")
	}

	var resTen tensor
	switch inPlace {
	case true:
		resTen = *t

	case false:
		resTen = tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}
	}

	otherData := *other.data()

	shape := t.Shape()
	otherShape := other.Shape()

	for i := range shape[0] {
		for j := range shape[1] {
			resTen.Data[i*shape[1]+j] = t.Data[i*shape[1]+j] + otherData[i%otherShape[0]*otherShape[1]+j%otherShape[1]]
		}
	}

	return &resTen, nil
}

func (t *tensor) Add(other Tensor, inPlace bool) (Tensor, error) {
	if !t.Shape().Eq(other.Shape()) {
		return nil, errors.New("Dimensions not the same")
	}

	var resTen tensor
	if inPlace {
		resTen = *t
	} else {
		resTen = tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}
	}

	otherData := other.data()
	for i := 0; i < len(t.Data); i++ {
		resTen.Data[i] = t.Data[i] + (*otherData)[i]
	}

	return &resTen, nil
}

func (t *tensor) Subtract(other Tensor, inPlace bool) (Tensor, error) {
	if !t.Shape().Eq(other.Shape()) {
		return nil, errors.New("Dimensions not the same")
	}

	var resTen tensor
	if inPlace {
		resTen = *t
	} else {
		resTen = tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}
	}

	otherData := other.data()
	for i := 0; i < len(t.Data); i++ {
		resTen.Data[i] = t.Data[i] - (*otherData)[i]
	}

	return &resTen, nil
}

func (t *tensor) Multiply(other Tensor, inPlace bool) (Tensor, error) {
	if !t.Shape().Eq(other.Shape()) {
		return nil, errors.New("Dimensions not the same")
	}

	var resTen tensor
	if inPlace {
		resTen = *t
	} else {
		resTen = tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}
	}

	otherData := other.data()
	for i := 0; i < len(t.Data); i++ {
		resTen.Data[i] = t.Data[i] * (*otherData)[i]
	}

	return &resTen, nil
}

func (t *tensor) Divide(other Tensor, inPlace bool) (Tensor, error) {
	if !t.Shape().Eq(other.Shape()) {
		return nil, errors.New("Dimensions not the same")
	}

	var resTen tensor
	if inPlace {
		resTen = *t
	} else {
		resTen = tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}
	}

	otherData := other.data()
	for i := 0; i < len(t.Data); i++ {
		otherVal := (*otherData)[i]
		if otherVal == 0 {
			return nil, errors.New("Cannot divide by zero")
		}
		resTen.Data[i] = t.Data[i] + (*otherData)[i]
	}

	return &resTen, nil
}

func (t *tensor) Map(fn func(float64) (float64, error), inPlace bool) (Tensor, error) {
	var resTen tensor
	if inPlace {
		resTen = *t
	} else {
		resTen = tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}
	}

	for i := 0; i < len(t.Data); i++ {
		y, err := fn(t.Data[i])
		if err != nil {
			return nil, err
		}
		resTen.Data[i] = y
	}

	return &resTen, nil
}

func (t *tensor) MapBatch(fn func(...float64) (float64, error), inPlace bool, others ...Tensor) (Tensor, error) {
	for _, other := range others {

		if !t.Shape().Eq(other.Shape()) {
			return nil, errors.New("Shapes do not match")
		}
	}

	var resTen tensor
	switch inPlace {
	case true:
		resTen = *t

	case false:
		resTen = tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}
	}

	for i := 0; i < len(t.Data); i++ {

		inputs := []float64{t.Data[i]}
		for _, other := range others {
			inputs = append(inputs, (*other.data())[i])
		}

		y, err := fn(inputs...)
		if err != nil {
			return nil, err
		}

		resTen.Data[i] = y
	}

	return &resTen, nil
}

func (t *tensor) ScalarAdd(x float64, inPlace bool) Tensor {
	if inPlace {
		for i := 0; i < len(t.Data); i++ {
			t.Data[i] += x
		}
		return nil
	}
	resTen := tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}

	for i := 0; i < len(t.Data); i++ {
		resTen.Data[i] = t.Data[i] + x
	}
	return &resTen
}

func (t *tensor) ScalarSubtract(x float64, inPlace bool) Tensor {
	if inPlace {
		for i := 0; i < len(t.Data); i++ {
			t.Data[i] -= x
		}
		return nil
	}
	resTen := tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}

	for i := 0; i < len(t.Data); i++ {
		resTen.Data[i] = t.Data[i] - x
	}
	return &resTen
}

func (t *tensor) ScalarMultiply(x float64, inPlace bool) Tensor {
	if inPlace {
		for i := 0; i < len(t.Data); i++ {
			t.Data[i] *= x
		}
		return nil
	}
	resTen := tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}

	for i := 0; i < len(t.Data); i++ {
		resTen.Data[i] = t.Data[i] * x
	}
	return &resTen
}

func (t *tensor) ScalarDivide(x float64, inPlace bool) (Tensor, error) {
	if x == 0 {
		return nil, errors.New("cannot divide by zero")
	}

	if inPlace {
		for i := 0; i < len(t.Data); i++ {
			t.Data[i] /= x
		}
		return nil, nil
	}
	resTen := tensor{TShape: t.Shape().Clone(), Data: make([]float64, len(t.Data))}

	for i := 0; i < len(t.Data); i++ {
		resTen.Data[i] = t.Data[i] - x
	}
	return &resTen, nil
}

func (t *tensor) Sum() float64 {
	if t.Data == nil || len(t.Data) == 0 {
		return 0.0
	}
	sum := 0.0
	for i := 0; i < len(t.Data); i++ {
		sum += t.Data[i]
	}
	return sum
}

func (t *tensor) Avg() float64 {
	if t.Data == nil || len(t.Data) == 0 {
		return 0.0
	}
	sum := 0.0
	for i := 0; i < len(t.Data); i++ {
		sum += t.Data[i]
	}
	return sum / float64(len(t.Data))
}

func (t *tensor) Min() float64 {
	if t.Data == nil || len(t.Data) == 0 {
		return math.Inf(-1)
	}
	min := t.Data[0]
	for i := 0; i < len(t.Data); i++ {
		if t.Data[i] > min {
			min = t.Data[i]
		}
	}

	return min
}

func (t *tensor) Max() float64 {
	if t.Data == nil || len(t.Data) == 0 {
		return math.Inf(1)
	}
	max := t.Data[0]
	for i := 0; i < len(t.Data); i++ {
		if t.Data[i] > max {
			max = t.Data[i]
		}
	}

	return max
}

func (t *tensor) Tile(reps []int) (Tensor, error) {

	return nil, nil
}

func (t *tensor) AxisSum(axis int) (Tensor, error) {
	if !t.Shape().IsMatrix() {
		return nil, errors.New("AxisSum() only implemented for matrices")
	}

	var resVec tensor
	switch axis {
	case 0:
		cols, _ := t.Shape().DimSize(1)
		resVec = tensor{TShape: []int{1, cols}, Data: make([]float64, cols)}
		t.Transpose(true)

	case 1:
		rows, _ := t.Shape().DimSize(0)
		resVec = tensor{TShape: []int{rows, 1}, Data: make([]float64, rows)}

	default:
		return nil, errors.New("Invalid axis choice chooses 0: for the sum of all the columns, and 1: for the sum of all the rows")
	}

	cols, _ := t.Shape().DimSize(1)
	for i := 0; i < t.Size(); i++ {
		resVec.Data[i/cols] += t.Data[i]
	}

	if axis == 0 {
		t.Transpose(true)
	}

	return &resVec, nil
}

func NewTensor(shape []int, input []float64) Tensor {
	return &tensor{TShape: shape, Data: input}
}

func (t *tensor) Shape() Shape {
	return t.TShape
}

func (t *tensor) Strides() []int {
	return t.TShape.CalcStrides()
}

func (t *tensor) Dims() int {
	return t.TShape.Dims()
}

func (t *tensor) Size() int {
	return t.TShape.TotalSize()
}

func (t *tensor) data() *[]float64 {
	return &t.Data
}

func Multiply(t_1, t_2 Tensor, inPlace bool) (Tensor, error) {
	t1, ok := t_1.(*tensor)
	if !ok {
		return nil, errors.New("Invalid first tensor")
	}

	t2, ok := t_2.(*tensor)
	if !ok {
		return nil, errors.New("Invalid second tensor")
	}

	if !t1.Shape().Eq(t2.Shape()) {
		return nil, errors.New("Shapes do not match")
	}

	resTen := tensor{TShape: t1.Shape().Clone(), Data: make([]float64, t1.Size())}
	for i := 0; i < len(t1.Data); i++ {
		resTen.Data[i] = (t1.Data)[i] * (t2.Data)[i]
	}

	return &resTen, nil
}

func Divide(t_1, t_2 Tensor, inPlace bool) (Tensor, error) {
	t1, ok := t_1.(*tensor)
	if !ok {
		return nil, errors.New("Invalid first tensor")
	}

	t2, ok := t_2.(*tensor)
	if !ok {
		return nil, errors.New("Invalid second tensor")
	}

	if !t1.Shape().Eq(t2.Shape()) {
		return nil, errors.New("Tensors not the same shape")
	}

	if inPlace {
		for i := 0; i < len(t1.Data); i++ {
			if t2.Data[i] == 0 {
				return nil, errors.New("Cannot divide by zero")
			}
			t1.Data[i] /= t2.Data[i]
		}
		return t1, nil
	}

	resTen := tensor{TShape: t1.Shape().Clone(), Data: make([]float64, t1.Size())}
	for i := 0; i < len(t1.Data); i++ {
		if t2.Data[i] == 0 {
			return nil, errors.New("Cannot divide by zero")
		}
		resTen.Data[i] = t1.Data[i] / t2.Data[i]
	}

	return &resTen, nil
}

func (t *tensor) MatMul(other Tensor) (Tensor, error) {
	if !t.Shape().IsMatrix() || !other.Shape().IsMatrix() {
		return nil, errors.New("Not both matrices")
	}

	if t.Shape()[1] != other.Shape()[0] {
		return nil, errors.New("matrices dimensions not compatible for matrix multiplication")
	}

	tData := t.data()
	oData := other.data()

	m := t.Shape()[0]     // Height of first matrix
	n := t.Shape()[1]     // Width of first matrix, height of second matrix
	p := other.Shape()[1] // Width of second matrix

	resTen := tensor{TShape: []int{m, p}, Data: make([]float64, m*p)}

	var wg sync.WaitGroup
	for i := 0; i < m; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < p; j++ {
				sum := 0.0
				for k := 0; k < n; k++ {
					sum += (*tData)[i*n+k] * (*oData)[k*p+j]
				}
				resTen.Data[i*p+j] = sum
			}
		}(i)
	}
	wg.Wait()

	return &resTen, nil
}

func (t *tensor) BatchMatMul(other Tensor) (Tensor, error) {
	if t.Dims() != 3 || !other.Shape().IsMatrix() {
		return nil, errors.New("Incorrect dimensions for batch matrix multiplication")
	}

	if t.Shape()[2] != other.Shape()[0] {
		return nil, errors.New("Incorrect height and widht dimension for matrix multiplication")
	}

	tData := t.data()
	oData := other.data()

	batches := t.Shape()[0] // Amount of batches in first tensor
	m := t.Shape()[1]       // Height of first tensor
	n := t.Shape()[2]       // Width of first tensor, height of second tensor
	p := other.Shape()[1]   // Width of second tensor

	resTen := tensor{TShape: []int{batches, m, p}, Data: make([]float64, batches*m*p)}

	for b := 0; b < batches; b++ { // Batch in tensor
		oldB := b * m * n
		newB := b * m * p
		for i := 0; i < m; i++ { // Row in  tensor
			for j := 0; j < p; j++ { // Column in matrix
				sum := 0.0
				for k := 0; k < n; k++ {
					sum += (*tData)[oldB+i*n+k] * (*oData)[k*p+j]
				}
				resTen.Data[newB+i*p+j] = sum
			}
		}
	}

	return &resTen, nil
}

func Add(t_1, t_2 Tensor, inPlace bool) (Tensor, error) {
	return nil, nil

}
