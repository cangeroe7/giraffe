package main

import (
	"fmt"
	u "github.com/cangeroe7/giraffe/internal/utils"
	l "github.com/cangeroe7/giraffe/pgk/layers"
	m "github.com/cangeroe7/giraffe/pgk/model"
	o "github.com/cangeroe7/giraffe/pgk/optimizers"
)

func main() {
	model := m.Sequential()

	model.Add(l.Dense(8, "relu"))
	model.Add(l.Dense(8, "relu"))
	model.Add(l.Dense(1, "sigmoid"))

	xTrain, yTrain, err := u.LoadCSV("/home/tskraan/giraffe/data/breastcancer/data.csv")
	columns := len(xTrain[0])

	shape := []int{1, columns}

	loss := "bce"
	optimizer := o.SGD{LearningRate: 0.01}

	metrics := []string{"accuracy, loss"}
	model.Compile(shape, loss, &optimizer, metrics)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}

	err = model.Fit(xTrain, yTrain, 32, 100, true)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		fmt.Println("fitting messed up")
	}

	model.History()
	// should be 1
	var testCase = [][]float64{{15.85, 23.95, 103.7, 782.7, 0.08401, 0.1002, 0.09938, 0.05364, 0.1847, 0.05338, 0.4033, 1.078, 2.903, 36.58, 0.009769, 0.03126, 0.05051, 0.01992, 0.02981, 0.003002, 16.84, 27.66, 112, 876.5, 0.1131, 0.1924, 0.2322, 0.1119, 0.2809, 0.06287}}

	test, err := model.Evaluate(&testCase)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		fmt.Println("evaluate messed up")
	}

	fmt.Printf("test: %v\n", test)
}
