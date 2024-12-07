package tensor

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
)

type Number interface {
  int | float64
}

type Tensor interface {
	// Info about the ndarray
	Shape() Shape
	Strides() []int
	Dims() int
	Size() int

	// Data Access
	data() *[]float64
	DataCopy() []float64
	ValueAt(index int) float64
	SetValueAt(index int, value float64) error
  AddValueAt(index int, value float64) error

	// Stuff Like Transpose
	Transpose(inPlace bool) Tensor
	Reshape(shape Shape) error

	// Mapping functions
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

	MaxIndex() int
	MinIndex() int
	AvgIndex() int

	AxisSum(axis int) (Tensor, error)

	Pad(pads ...int) (Tensor, error)
	Tile(reps []int) (Tensor, error)

	// Linear Operations
	MatMul(other Tensor) (Tensor, error)

	CrossCorrelate(other Tensor, strides [2]int, resTen Tensor) (Tensor, error)

	RegionSlice(startRow, startCol, numRows, numCols int) (Tensor, []int, error)

	Print()
}

func (t *tensor) Print() {

	tIter, _ := IterFromTensor(t, "")

	for ten, ok := tIter.Next(); ok; ten, ok = tIter.Next() {

		rows := ten.Shape().Rows()
		cols := ten.Shape().Cols()

		tData := *ten.data()
		fmt.Printf("\n")
		for i := range rows {
			fmt.Printf("\n")
			for j := range cols {
				fmt.Printf("%v ", tData[i*cols+j])
			}
		}
	}
}

type tensor struct {
	TShape Shape
	Data   []float64
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

func (t *tensor) CrossCorrelate(kernels Tensor, strides [2]int, resTen Tensor) (Tensor, error) {
	if t.Shape().Channels() != kernels.Shape().Channels() {
		return nil, errors.New("input doesn't have the same amount of channels as filters")
	}

	kRows, kCols := kernels.Shape().Rows(), kernels.Shape().Cols()

	outRows := (t.Shape().Rows()-kRows)/strides[0] + 1
	outCols := (t.Shape().Cols()-kCols)/strides[1] + 1

	outShape := []int{outRows, outCols}

	if resTen == nil {
		resTen = ZerosTensor([]int{outRows, outCols})
	} else {
		if !resTen.Shape().Eq(outShape) {
			return nil, errors.New("resTen doesn't have right dimensions to put output of cross correlation in")
		}
	}

	kernelIter, err := IterFromTensor(kernels, "mat")
	if err != nil {
		return nil, err
	}

	matIter, err := IterFromTensor(t, "mat")
	if err != nil {
		return nil, err
	}

	for mat, ok := matIter.Next(); ok; mat, ok = matIter.Next() {
		kernel, ok := kernelIter.Next()
		if !ok {
			return nil, errors.New("this isn't supposed to happen")
		}

		matData := *mat.data()
		kernelData := *kernel.data()
		resData := *resTen.data()

		for i := 0; i < outRows; i++ {
			for j := 0; j < outCols; j++ {
				sum := 0.0

				// Calculate the kernel
				for ki := 0; ki < kRows; ki++ {
					for kj := 0; kj < kCols; kj++ {
						matRow := i*strides[0] + ki
						matCol := j*strides[1] + kj
						matIndex := matRow*outCols + matCol
						kIndex := ki*kCols + kj
						sum += matData[matIndex] * kernelData[kIndex]
					}
				}
				resData[i*outCols+j] += sum
			}
		}
	}

	return resTen, nil
}

func (t *tensor) RegionSlice(startRow, startCol, numRows, numCols int) (Tensor, []int, error) {

	if !t.Shape().IsMatrix() {
		return nil, nil, errors.New("tensor must be a matrix")
	}

	if startRow < 0 || startCol < 0 || numRows <= 0 || numCols <= 0 {
		return nil, nil, errors.New("start indices must be non-negative and size must be positive")
	}

	if startRow+numRows > t.Shape().Rows() || startCol+numCols > t.Shape().Cols() {
		return nil, nil, errors.New("region exceeds matrix bounds")
	}

	data := make([]float64, 0, numRows*numCols)
  indeces := make([]int, 0, numRows*numCols)

	for i := 0; i < numRows; i++ {

		startIdx := (startRow+i)*t.Shape().Cols() + startCol
		endIdx := startIdx + numCols

		for i := startIdx; i < endIdx; i++ {
			data = append(data, t.ValueAt(i))
      indeces = append(indeces, i)
		}
	}

	// Create the new tensor
	resMat, _ := TensorFrom([]int{numRows, numCols}, data)

	return resMat, indeces, nil
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

func (t *tensor) Reshape(shape Shape) error {
	if t.Shape().TotalSize() != shape.TotalSize() {
		return errors.New("Cannot reshape when totalSizes are not the same")
	}

	t.TShape = shape

	return nil
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

func (t *tensor) MaxIndex() int {
	if t.Data == nil || len(t.Data) == 0 {
		return -1
	}
	max := t.Data[0]
  maxIdx := 0
	for i := 0; i < len(t.Data); i++ {
		if t.Data[i] > max {
			max = t.Data[i]
      maxIdx = i
		}
	}

	return maxIdx
}
func (t *tensor) MinIndex() int {
	if t.Data == nil || len(t.Data) == 0 {
		return -1
	}
	min := t.Data[0]
  minIdx := 0
	for i := 0; i > len(t.Data); i++ {
		if t.Data[i] > min {
			min = t.Data[i]
      minIdx = i
		}
	}

	return minIdx
}

func (t *tensor) AvgIndex() int {
	if t.Data == nil || len(t.Data) == 0 {
		return -1
	}
	avg := t.Data[0]
  avgIdx := 0
	for i := 0; i < len(t.Data); i++ {
		if t.Data[i] > avg {
			avg = t.Data[i]
      avgIdx = i
		}
	}

	return avgIdx
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

func (t *tensor) Pad(pads ...int) (Tensor, error) {
	for _, pad := range pads {
		if pad < 0 {
			return nil, errors.New("negative padding value given")
		}
	}

	var T, R, B, L int
	switch len(pads) {
	case 1:
		T, B, L, R = pads[0], pads[0], pads[0], pads[0]

	case 2:
		T, B, L, R = pads[0], pads[0], pads[1], pads[1]

	case 3:
		T, L, R, B = pads[0], pads[1], pads[1], pads[2]

	case 4:
		T, R, B, L = pads[0], pads[1], pads[2], pads[3]

	default:
		return t, errors.New("Incorrect amount of padding vals given")
	}

	if T+R+B+L == 0 {
		return t, nil
	}

	paddedShape := t.Shape().Clone()
	paddedShape[len(paddedShape)-1] += L + R
	paddedShape[len(paddedShape)-2] += T + B

	resTen := ZerosTensor(paddedShape)
	resTenIter, err := IterFromTensor(resTen, "")
	if err != nil {
		return nil, err
	}

	tenIter, err := IterFromTensor(t, "")
	if err != nil {
		return nil, err
	}

	for mat, ok := tenIter.Next(); ok; mat, ok = tenIter.Next() {
		resMat, ok := resTenIter.Next()
		if !ok {
			return nil, errors.New("Shouldn't be possible, but result matrix iterator ran out of matrices before tensor")
		}

		Data := *mat.data()
		resTenData := *resMat.data()
		for i := 0; i < mat.Shape().Rows(); i++ {

			for j := 0; j < mat.Shape().Cols(); j++ {
				resTenData[(i+T)*resMat.Shape().Cols()+j+L] = Data[i*mat.Shape().Cols()+j]
			}
		}
	}

	return resTen, nil
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

func (t *tensor) DataCopy() []float64 {
	resArr := make([]float64, len(t.Data))
	copy(resArr, t.Data)
	return resArr
}

func (t *tensor) ValueAt(index int) float64 {
	return t.Data[index]
}

func (t *tensor) SetValueAt(index int, value float64) error {
	if index >= t.Shape().TotalSize() {
		return errors.New("index out of range")
	}

	t.Data[index] = value

	return nil
}

func (t *tensor) AddValueAt(index int, value float64) error {
	if index >= t.Shape().TotalSize() {
		return errors.New("index out of range")
	}

	t.Data[index] += value

	return nil
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

	m := t.Shape().Clone()[0]     // Height of first matrix
	n := t.Shape().Clone()[1]     // Width of first matrix, height of second matrix
	p := other.Shape().Clone()[1] // Width of second matrix

	resTen := tensor{TShape: []int{m, p}, Data: make([]float64, m*p)}

	other.Transpose(true)

	var wg sync.WaitGroup
	for i := range m {
		IxP := i * p
		IxN := i * n
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := range p {
				JxN := j * n
				sum := 0.0
				for k := range n {
					sum += (*tData)[IxN+k] * (*oData)[JxN+k]
				}
				resTen.Data[IxP+j] = sum
			}
		}(i)
	}
	wg.Wait()

	other.Transpose(true)

	return &resTen, nil
}

func Add(t_1, t_2 Tensor, inPlace bool) (Tensor, error) {
	return nil, nil

}
