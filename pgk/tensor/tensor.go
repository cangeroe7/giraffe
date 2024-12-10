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
	RepMultiply(other Tensor, inPlace bool) (Tensor, error)
	Add(other Tensor, inPlace bool) (Tensor, error)
	Subtract(other Tensor, inPlace bool) (Tensor, error)
	Multiply(other Tensor, inPlace bool) (Tensor, error)
	Divide(other Tensor, inPlace bool) (Tensor, error)
	AddMatrix(other Tensor) error

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

	ArgMax(axis int) ([]int, error)

	AxisSum(axis int) (Tensor, error)

	Pad(pads ...int) (Tensor, error)
	Trim(trim_sizes ...int) (Tensor, error)
	Tile(reps_rows, reps_cols int) (Tensor, error)
	Dilate(rows, cols int) (Tensor, error)
	Normalize() error
	OneHotEncode(size int) (Tensor, error)

	// Linear Operations
	MatMul(other Tensor) (Tensor, error)

	CrossCorrelate(kernels Tensor, strides [2]int, resTen Tensor) (Tensor, error)
	Convolve(kernels Tensor, strides [2]int, resTen Tensor) (Tensor, error)

	Slice(start, end int) (Tensor, error)
	RegionSlice(startRow, startCol, numRows, numCols int) (Tensor, []int, error)
	BatchSlice(startBatch, endBatch int) (Tensor, error)

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

	resTen := ZerosTensor([]int{n, n})
	for i := 0; i < n*n; i += n + 1 {
		resTen.SetValueAt(i, 1.0)
	}

	return resTen, nil
}

func (t *tensor) CrossCorrelate(kernels Tensor, strides [2]int, resTen Tensor) (Tensor, error) {
	// Check that the input tensor and kernels have the same number of channels
	if t.Shape().Channels() != kernels.Shape().Channels() {
		return nil, errors.New("input doesn't have the same amount of channels as filters")
	}

	kRows, kCols := kernels.Shape().Rows(), kernels.Shape().Cols()

	// Calculate output dimensions
	outRows := (t.Shape().Rows()-kRows)/strides[0] + 1
	outCols := (t.Shape().Cols()-kCols)/strides[1] + 1

	// Create or check result tensor dimensions
	outShape := []int{outRows, outCols}
	if resTen == nil {
		resTen = ZerosTensor(outShape)
	} else if !resTen.Shape().Eq(outShape) {
		return nil, errors.New("resTen doesn't have right dimensions to hold cross correlation result")
	}

	// Access tensor data references
	matData := *t.data()
	kernelData := *kernels.data()
	resData := *resTen.data()

	// Iterate through the input tensor
	for i := 0; i < outRows; i++ {
		for j := 0; j < outCols; j++ {
			sum := 0.0

			// Cross-correlation operation
			for ki := 0; ki < kRows; ki++ {
				for kj := 0; kj < kCols; kj++ {
					matRow := i*strides[0] + ki
					matCol := j*strides[1] + kj
					matIndex := matRow*t.Shape().Cols() + matCol
					kIndex := ki*kCols + kj
					sum += matData[matIndex] * kernelData[kIndex]
				}
			}

			// Assign the result
			resData[i*outCols+j] += sum
		}
	}

	return resTen, nil
}

func (t *tensor) Convolve(kernels Tensor, strides [2]int, resTen Tensor) (Tensor, error) {
	// Check that the input tensor and kernels have the same number of channels
	if t.Shape().Channels() != kernels.Shape().Channels() {
		return nil, errors.New("input doesn't have the same amount of channels as filters")
	}

	kRows, kCols := kernels.Shape().Rows(), kernels.Shape().Cols()

	// Calculate output dimensions
	outRows := (t.Shape().Rows()-kRows)/strides[0] + 1
	outCols := (t.Shape().Cols()-kCols)/strides[1] + 1

	// Create or check result tensor dimensions
	outShape := []int{outRows, outCols}
	if resTen == nil {
		resTen = ZerosTensor(outShape)
	} else if !resTen.Shape().Eq(outShape) {
		return nil, errors.New("resTen doesn't have right dimensions to hold convolution result")
	}

	// Access tensor data references
	matData := *t.data()
	kernelData := *kernels.data()
	resData := *resTen.data()

	// Iterate through the input tensor
	for i := 0; i < outRows; i++ {
		for j := 0; j < outCols; j++ {
			sum := 0.0

			// Convolution operation (kernel is flipped)
			for ki := 0; ki < kRows; ki++ {
				for kj := 0; kj < kCols; kj++ {
					matRow := i*strides[0] + ki
					matCol := j*strides[1] + kj
					matIndex := matRow*t.Shape().Cols() + matCol

					// Kernel is flipped for convolution
					kIndex := (kRows-1-ki)*kCols + (kCols - 1 - kj)
					sum += matData[matIndex] * kernelData[kIndex]
				}
			}
			// Assign the result
			resData[i*outCols+j] += sum
		}
	}

	return resTen, nil
}

func (t *tensor) Slice(start, end int) (Tensor, error) {
	switch {
	case start >= end:
		return nil, errors.New("batchslice: end is less than or equal to start")

	case start < 0 || end < 0:
		return nil, errors.New("batchslice: points cannot be negative")

	case end >= t.Shape().TotalSize():
		return nil, errors.New("batchslice: end point out of range")
	}

	data := t.Data[start:end]

	resTen, err := TensorFrom([]int{1, len(data)}, data)
	if err != nil {
		return nil, err
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

func (t *tensor) BatchSlice(startBatch, endBatch int) (Tensor, error) {
	switch {
	case startBatch >= endBatch:
		return nil, errors.New("end is less than or equal to start")

	case startBatch < 0 || endBatch < 0:
		return nil, errors.New("points cannot be negative")

  
	case endBatch >= t.Shape().Batches():
		return nil, errors.New("end point out of range")
	}

	startIdx := startBatch * t.Strides()[0]
	endIdx := endBatch * t.Strides()[0]

	newShape := []int{endBatch - startBatch, t.Shape().Channels(), t.Shape().Rows(), t.Shape().Cols()}

	newData := t.Data[startIdx:endIdx]

	resTen, err := TensorFrom(newShape, newData)
	if err != nil {
		return nil, err
	}

	return resTen, nil
}
func (t *tensor) Transpose(inPlace bool) Tensor {

	matrixIter, _ := IterFromTensor(t, "matrix")

	result := make([]float64, 0, t.Shape().TotalSize())
	for matrix, ok := matrixIter.Next(); ok; matrix, ok = matrixIter.Next() {
		result = append(result, subTranspose(matrix.data(), t.Shape().Rows(), t.Shape().Cols())...)
	}

	newShape := t.Shape().Clone()
	newShape[len(newShape)-2], newShape[len(newShape)-1] = newShape[len(newShape)-1], newShape[len(newShape)-2]

	if inPlace {
		t.Data = result
		t.Reshape(newShape)
		return t
	}

	resTen, _ := TensorFrom(newShape, result)

	return resTen
}

func subTranspose(matrix *[]float64, rows, cols int) []float64 {

	transposed := make([]float64, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j*rows+i] = (*matrix)[i*cols+j]
		}
	}

	return transposed
}

func (t *tensor) Reshape(shape Shape) error {
	for _, val := range shape {
		if val <= 0 {
			return errors.New("Dimensions cannot be negative")
		}
	}

	if t.Shape().TotalSize()%shape.TotalSize() != 0 {
		return errors.New("Cannot reshape when totalSizes are not the same")
	}
	switch len(shape) {
	case 0, 1:
		return errors.New("must have atleast a matrix input")

	case 2:
		t.TShape = []int{t.Shape().TotalSize() / shape.TotalSize(), 1, shape[0], shape[1]}

	case 3:
		t.TShape = []int{t.Shape().TotalSize() / shape.TotalSize(), shape[0], shape[1], shape[2]}

	case 4:
		t.TShape = shape

	default:
		return errors.New("Invalid shape length")
	}

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

	bats, chas, rows, cols := shape.Batches(), shape.Channels(), shape.Rows(), shape.Cols()
	oBats, oChs, oRows, oCols := otherShape.Batches(), otherShape.Channels(), otherShape.Rows(), otherShape.Cols()

	for b := range bats {
		resBatch := b * chas * rows * cols
		otherBatch := (b % oBats) * oChs * oRows * oCols
		for c := range chas {
			resChannel := c * rows * cols
			otherChannel := (c % oChs) * oRows * oCols
			for i := range rows {
				resRow := i * cols
				otherRow := (i % oRows) * oCols
				for j := range cols {
					resIdx := resBatch + resChannel + resRow + j
					otherIdx := otherBatch + otherChannel + otherRow + j%oCols
					resTen.Data[resIdx] = t.Data[resIdx] + otherData[otherIdx]
				}
			}
		}
	}

	return &resTen, nil
}

func (t *tensor) RepMultiply(other Tensor, inPlace bool) (Tensor, error) {

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

	for i := range shape.Rows() {
		for j := range shape.Cols() {
			resTen.Data[i*shape.Cols()+j] = t.Data[i*shape.Cols()+j] * otherData[i%otherShape.Rows()*otherShape.Cols()+j%otherShape.Cols()]
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

func (t *tensor) AddMatrix(other Tensor) error {
	if !other.Shape().IsMatrix() {
		return errors.New("cannot add non matrix")
	}

	if len(t.Shape()) != 4 {
		return errors.New("cannot add if tensor doesn't ahve batches")
	}

	if other.Shape().TotalSize() != t.TShape.Rows()*t.TShape.Cols() {
		return errors.New("matrix sizes not the same size")
	}

	t.Data = append(t.Data, *other.data()...)

	t.TShape[0] += 1

	return nil
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
		if t.Data[i] < min {
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

func (t *tensor) ArgMax(axis int) ([]int, error) {
	if axis > 1 || axis < 0 {
		return nil, errors.New("Axis not 1 or 0")
	}

	if axis == 0 {
		t.Transpose(true)
	}

	result := make([]int, 0, t.Shape().Rows())

	rowIter, err := IterFromTensor(t, "rows")
	if err != nil {
		return nil, err
	}

	for row, ok := rowIter.Next(); ok; row, ok = rowIter.Next() {
		maxIndex := row.MaxIndex()
		result = append(result, maxIndex)
	}

	if axis == 0 {
		t.Transpose(true)
	}

	return result, nil
}

func (t *tensor) Tile(reps_rows, reps_cols int) (Tensor, error) {
	if reps_rows == 0 || reps_cols == 0 {
		return t, nil
	}

	newShape := t.Shape().Clone()
	newShape[len(newShape)-2] *= reps_rows
	newShape[len(newShape)-1] *= reps_cols

	resTen := ZerosTensor(newShape)

	resMatrixIter, err := IterFromTensor(resTen, "matrix")
	if err != nil {
		return nil, err
	}

	inputMatrixIter, err := IterFromTensor(t, "matrix")
	if err != nil {
		return nil, err
	}

	for resMatrix, ok := resMatrixIter.Next(); ok; resMatrix, ok = resMatrixIter.Next() {
		inputMatrix, _ := inputMatrixIter.Next()

		inputData := *inputMatrix.data()
		resData := *resMatrix.data()

		for i := range newShape.Rows() {
			for j := range newShape.Cols() {
				inputRow := i % inputMatrix.Shape().Rows()
				inputCol := j % inputMatrix.Shape().Cols()
				inputIdx := inputRow*inputMatrix.Shape().Cols() + inputCol

				resData[i*newShape.Cols()+j] = inputData[inputIdx]
			}
		}
	}

	return resTen, nil
}

func (t *tensor) Dilate(rows, cols int) (Tensor, error) {

	if rows < 0 || cols < 0 {
		return nil, errors.New("dilate: dimensions cannot be negative")
	}

	if rows == 0 && cols == 0 {
		return t, nil
	}

	newHeight := t.Shape().Rows()*(rows+1) - rows
	newWidth := t.Shape().Cols()*(cols+1) - cols
	newShape := []int{t.Shape().Batches(), t.Shape().Channels(), newHeight, newWidth}

	resTen := ZerosTensor(newShape)

	resMatIter, err := IterFromTensor(resTen, "matrix")
	if err != nil {
		return nil, err
	}

	dataMatIter, err := IterFromTensor(t, "matrix")
	if err != nil {
		return nil, err
	}

	for dataMat, ok := dataMatIter.Next(); ok; dataMat, ok = dataMatIter.Next() {
		resMat, _ := resMatIter.Next()

		resMatData := *(resMat.data())
		for i := range dataMat.Shape().Rows() {
			for j := range dataMat.Shape().Cols() {
				resMatData[i*resMat.Shape().Cols()*rows+j*cols] = t.Data[i*dataMat.Shape().Cols()+j]
			}
		}

	}

	return resTen, nil
}

func (t *tensor) Normalize() error {

	t.Transpose(true)

	rowIter, err := IterFromTensor(t, "rows")
	if err != nil {
		return err
	}

	for row, ok := rowIter.Next(); ok; row, ok = rowIter.Next() {
		minVal := row.Min()
		maxVal := row.Max()

		if minVal == maxVal {
			row.ScalarMultiply(0.0, true)
			continue
		}

		row.ScalarSubtract(minVal, true)
		row.ScalarDivide(maxVal-minVal, true)
	}

	t.Transpose(true)

	return nil
}

func (t *tensor) OneHotEncode(size int) (Tensor, error) {
	resTen := ZerosTensor([]int{t.Shape().Rows(), size})
	for i, val := range t.Data {
		resTen.SetValueAt(i*size+int(val), 1.0)
	}

	return resTen, nil
}
func (t *tensor) AxisSum(axis int) (Tensor, error) {
	if !t.Shape().IsMatrix() {
		return nil, errors.New("AxisSum() only implemented for matrices")
	}

	var resVec tensor
	switch axis {
	case 0:
		cols := t.Shape().Cols()
		resVec = tensor{TShape: []int{1, cols}, Data: make([]float64, cols)}
		t.Transpose(true)

	case 1:
		rows := t.Shape().Rows()
		resVec = tensor{TShape: []int{rows, 1}, Data: make([]float64, rows)}

	default:
		return nil, errors.New("Invalid axis choice chooses 0: for the sum of all the columns, and 1: for the sum of all the rows")
	}

	cols := t.Shape().Cols()
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

func (t *tensor) Trim(trim_sizes ...int) (Tensor, error) {

	for _, trim := range trim_sizes {
		if trim < 0 {
			return nil, errors.New("negative padding value given")
		}
	}

	var T, R, B, L int
	switch len(trim_sizes) {
	case 1:
		T, B, L, R = trim_sizes[0], trim_sizes[0], trim_sizes[0], trim_sizes[0]

	case 2:
		T, B, L, R = trim_sizes[0], trim_sizes[0], trim_sizes[1], trim_sizes[1]

	case 3:
		T, L, R, B = trim_sizes[0], trim_sizes[1], trim_sizes[1], trim_sizes[2]

	case 4:
		T, R, B, L = trim_sizes[0], trim_sizes[1], trim_sizes[2], trim_sizes[3]

	default:
		return t, errors.New("Incorrect amount of padding vals given")
	}

	if T+R+B+L == 0 {
		return t, nil
	}

	trimmedShape := t.Shape().Clone()
	trimmedShape[len(trimmedShape)-1] -= L + R
	trimmedShape[len(trimmedShape)-2] -= T + B

	if trimmedShape[len(trimmedShape)-1] <= 0 || trimmedShape[len(trimmedShape)-2] <= 0 {
		return nil, errors.New("Trimmed shape must stay positive")
	}

	resTen := ZerosTensor(trimmedShape)
	resMatrixIter, err := IterFromTensor(resTen, "channel")
	if err != nil {
		return nil, err
	}

	inputMatrixIter, err := IterFromTensor(t, "channel")
	if err != nil {
		return nil, err
	}

	for inputMatrix, ok := inputMatrixIter.Next(); ok; inputMatrix, ok = inputMatrixIter.Next() {
		resMatrix, ok := resMatrixIter.Next()
		if !ok {
			return nil, errors.New("Shouldn't be possible, but result matrix iterator ran out of matrices before tensor")
		}

		inputData := *inputMatrix.data()
		resData := *resMatrix.data()
		for i := T; i < resMatrix.Shape().Rows(); i++ {

			for j := L; j < resMatrix.Shape().Cols(); j++ {
				resData[(i+T)*resMatrix.Shape().Cols()+j+L] = inputData[i*inputMatrix.Shape().Cols()+j]
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

	if t.Shape().Cols() != other.Shape().Rows() {
		return nil, errors.New("matrices dimensions not compatible for matrix multiplication")
	}

	tData := t.data()
	oData := other.data()

	m := t.Shape().Clone().Rows()     // Height of first matrix
	n := t.Shape().Clone().Cols()     // Width of first matrix, height of second matrix
	p := other.Shape().Clone().Cols() // Width of second matrix

	resTen := ZerosTensor([]int{m, p})
	resTenData := *resTen.data()

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
				resTenData[IxP+j] = sum
			}
		}(i)
	}
	wg.Wait()

	other.Transpose(true)

	return resTen, nil
}

func Add(t_1, t_2 Tensor, inPlace bool) (Tensor, error) {
	return nil, nil

}

func Shuffle(x, y Tensor) error {
	n := x.Shape().Batches()
	for i := n - 1; i > 0; i-- {
		j := rand.Intn(i + 1)

		xBatchI, err := x.BatchSlice(i, i+1)
		if err != nil {
			return nil
		}

		xBatchJ, err := x.BatchSlice(j, j+1)
		if err != nil {
			return err
		}

		xBatchIData := xBatchI.DataCopy()
		copy(*xBatchI.data(), *xBatchJ.data())
		copy(*xBatchJ.data(), xBatchIData)

		yBatchI, err := y.BatchSlice(i, i+1)
		if err != nil {
			return nil
		}

		yBatchJ, err := y.BatchSlice(j, j+1)
		if err != nil {
			return err
		}

		yBatchIData := yBatchI.DataCopy()
		copy(*yBatchI.data(), *yBatchJ.data())
		copy(*yBatchJ.data(), yBatchIData)
	}
	return nil
}
