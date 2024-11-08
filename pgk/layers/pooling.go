package layers
//
//import (
	//"errors"
	//"math"
//)
//
//type Pooling struct {
	//poolSize [2]int
	//strides  [2]int
	//padding  bool
//}
//
//
//type MaxPooling2D struct {
	//p Pooling
//}
//
//type MinPooling2D struct {
	//p Pooling
//}
//
//type AvgPooling2D struct {
	//p Pooling
//}
//
//func (max *MaxPooling2D) operation(matrix *[][]float64, row, col int) (float64, error) {
	//numRows := len(*matrix)
	//if numRows == 0 {
		//return 0, errors.New("matrix is empty")
	//}
	//numCols := len((*matrix)[0])
//
	//if row < 0 || row+max.p.poolSize[0] > numRows || col < 0 || col+max.p.poolSize[1] > numCols {
		//return 0, errors.New("pool step out of range")
	//}
//
	//maxVal := (*matrix)[row][col]
//
	//for r := row; r < row+max.p.poolSize[0]; r++ {
		//for c := col; c < col+max.p.poolSize[1]; c++ {
			//newVal := (*matrix)[r][c]
			//if newVal > maxVal {
				//maxVal = newVal
			//}
		//}
	//}
//
	//return maxVal, nil
//}
//
//func (min *MinPooling2D) operation(matrix *[][]float64, row, col int) (float64, error) {
	//numRows := len(*matrix) // Get the number of rows in the matrix
	//if numRows == 0 {
		//return 0, errors.New("matrix is empty")
	//}
	//numCols := len((*matrix)[0]) // Get the number of columns in the matrix
//
	//// Check if the starting point (row, col) and pool size exceed matrix bounds
	//if row < 0 || row+min.p.poolSize[0] > numRows || col < 0 || col+min.p.poolSize[1] > numCols {
		//return 0, errors.New("pool step out of range")
	//}
//
	//minVal := (*matrix)[row][col]
//
	//for r := row; r < row+min.p.poolSize[0]; r++ {
		//for c := col; c < col+min.p.poolSize[1]; c++ {
			//newVal := (*matrix)[r][c]
			//if newVal < minVal {
				//minVal = newVal
			//}
		//}
	//}
	//return minVal, nil
//}
//
//func (avg *AvgPooling2D) operation(matrix *[][]float64, row, col int) (float64, error) {
	//numRows := len(*matrix) // Get the number of rows in the matrix
	//if numRows == 0 {
		//return 0, errors.New("matrix is empty")
	//}
	//numCols := len((*matrix)[0]) // Get the number of columns in the matrix
//
	//// Check if the starting point (row, col) and pool size exceed matrix bounds
	//if row < 0 || row+avg.p.poolSize[0] > numRows || col < 0 || col+avg.p.poolSize[1] > numCols {
		//return 0, errors.New("pool step out of range")
	//}
	//sum := 0.0
//
	//for r := row; r < row+avg.p.poolSize[0]; r++ {
		//for c := col; c < col+avg.p.poolSize[1]; c++ {
			//sum += (*matrix)[r][c]
		//}
	//}
//
	//avgVal := sum / (float64(avg.p.poolSize[0]) * float64(avg.p.poolSize[1]))
//
	//return avgVal, nil
//}
//
//func (max *MaxPooling2D) forward(input [][]float64) ([][]float64, error) {
	//output, err := max.p.process(input, max)
	//if err != nil {
		//return nil, err
	//}
	//return output, nil
//}
//
//func (min *MinPooling2D) forward(input [][]float64) ([][]float64, error) {
	//output, err := min.p.process(input, min)
	//if err != nil {
		//return nil, err
	//}
	//return output, nil
//}
//
//func (avg *AvgPooling2D) forward(input [][]float64) ([][]float64, error) {
	//output, err := avg.p.process(input, avg)
	//if err != nil {
		//return nil, err
	//}
	//return output, nil
//}
//
//type PoolingOperation struct {
//
//}
//
//func (max *MaxPooling2D) backward(gradient [][]float64) [][]float64 {
  //return gradient
//}
//
//func (min *MinPooling2D) backward(gradient [][]float64) [][]float64 {
  //return gradient
//}
//
//func (avg *AvgPooling2D) backward(gradient [][]float64) [][]float64 {
  //return gradient
//}
//
//func (p *Pooling) process(matrix [][]float64, op PoolingOperation) ([][]float64, error) {
//
	//outRows := int(math.Floor(float64(len(matrix)) / float64(p.strides[0])))
	//outCols := int(math.Floor(float64(len(matrix)) / float64(p.strides[1])))
//
	//if p.padding {
		//outRows := int(math.Ceil(float64(len(matrix)) / float64(p.strides[0])))
		//outCols := int(math.Ceil(float64(len(matrix)) / float64(p.strides[1])))
		//bottomPad := (outRows-1)*p.strides[0] + p.poolSize[0] - len(matrix)
		//rightPad := (outCols-1)*p.strides[1] + p.poolSize[1] - len(matrix[0])
		//paddedMat := make([][]float64, len(matrix)+bottomPad)
		//for i := range paddedMat {
			//paddedMat[i] = make([]float64, len(matrix[0])+rightPad)
		//}
//
		//for i := 0; i < len(matrix); i++ {
			//for j := 0; j < len(matrix[0]); j++ {
				//paddedMat[i][j] = matrix[i][j]
			//}
		//}
		//matrix = paddedMat
	//}
//
	//output := make([][]float64, outRows)
	//for i := 0; i < outRows; i++ {
		//output[i] = make([]float64, outCols)
	//}
//
	//var err error
	//for r := range outRows {
		//for c := range outCols {
			//output[r][c], err = op.operation(&matrix, r * p.strides[0], c * p.strides[1])
			//if err != nil {
				//return nil, err
			//}
		//}
	//}
//
	//return output, nil
//}
