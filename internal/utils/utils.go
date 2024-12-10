package utils

import (
	"encoding/csv"
	"os"
	"strconv"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

func LoadCSV(filePath string) (t.Tensor, t.Tensor, error) {
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

	var xTrain []float64
	var yTrain []float64

	for i, row := range rows {
		if i == 0 {
			continue
		}

		vals := len(row)

		for j := 1; j < vals; j++ {
			val, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return nil, nil, err
			}
			xTrain = append(xTrain, val)
		}

		correctVal, err := strconv.ParseFloat(row[0], 64)
		if err != nil {
			return nil, nil, err
		}
		yTrain = append(yTrain, correctVal)

	}

	xTrainShape := []int{len(rows)-1, len(rows[0])-1}
	yTrainShape := []int{len(rows)-1, 1}
  

	xTrainTen, err := t.TensorFrom(xTrainShape, xTrain)
  if err != nil {
    return nil, nil, err
  }
	yTrainTen, err := t.TensorFrom(yTrainShape, yTrain)
  if err != nil {
    return nil, nil, err
  }

	return xTrainTen, yTrainTen, nil
}
