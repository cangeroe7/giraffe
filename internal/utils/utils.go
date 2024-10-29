package utils

import (
	"encoding/csv"
	"math"
	"math/rand"
	"os"
	"strconv"
)

func Shuffle[T any, U any](x []T, y []U) {
	n := len(x)
	for i := n - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	}
}

type Normalizer struct {
	Activated bool
	minValues []float64
	maxValues []float64
}

func (n *Normalizer) SetFeatures(data *[][]float64) {
	rows := len(*data)
	if rows == 0 {
		return
	}
	cols := len((*data)[0])

	// Calculate min and max for each column
	minValues := make([]float64, cols)
	maxValues := make([]float64, cols)

	// Initialize min and max values
	for col := 0; col < cols; col++ {
		minValues[col] = math.MaxFloat64
		maxValues[col] = -math.MaxFloat64
	}

	// Find the min and max values for each column
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			value := (*data)[row][col]
			if value < minValues[col] {
				minValues[col] = value
			}
			if value > maxValues[col] {
				maxValues[col] = value
			}
		}
	}

	n.Activated = true
	n.minValues = minValues
	n.maxValues = maxValues
}

func (n *Normalizer) NormalizeData(data *[][]float64) {
	rows, cols := len(*data), len((*data)[0])
	if rows == 0 {
		return
	}

	// Normalize the data in place using min/max values
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			minVal := n.minValues[col]
			maxVal := n.maxValues[col]

			// Normalize each value in the dataset
			if maxVal != minVal {
				(*data)[row][col] = ((*data)[row][col] - minVal) / (maxVal - minVal)
			} else {
				// Avoid division by zero when max == min
				(*data)[row][col] = 0.0
			}
		}
	}
}

func LoadCSV(filePath string) ([][]float64, [][]float64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	var xTrain [][]float64
	var yTrain [][]float64

	for _, row := range rows {

		var inputRow []float64
		var correctAnswerRow []float64
		vals := len(row)

		for j := 1; j < vals; j++ {
			val, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return nil, nil, err
			}
			inputRow = append(inputRow, val)
		}

		correctVal, err := strconv.ParseFloat(row[0], 64)
		if err != nil {
			return nil, nil, err
		}
		correctAnswerRow = append(correctAnswerRow, correctVal)

		xTrain = append(xTrain, inputRow)
		yTrain = append(yTrain, correctAnswerRow)
	}
	return xTrain, yTrain, nil

}
