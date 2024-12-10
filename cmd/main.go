package main

import (
	"fmt"
	u "github.com/cangeroe7/giraffe/internal/utils"
	a "github.com/cangeroe7/giraffe/pgk/activations"
	l "github.com/cangeroe7/giraffe/pgk/layers"
	lo "github.com/cangeroe7/giraffe/pgk/losses"
	m "github.com/cangeroe7/giraffe/pgk/model"
	o "github.com/cangeroe7/giraffe/pgk/optimizers"
	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

func main() {

	err := testDenseBreastcancer()
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
}

func loadMNISTModelAndEvaluate() error {
	xTest, yTest, err := u.LoadCSV("/home/tskraan/giraffe/data/mnist/mnist_test_sample.csv")
	if err != nil {
		return err
	}

	shape := []int{28, 28}
	xTest.Reshape(shape)
	xTest.ScalarDivide(255.0, true)
	yTest, _ = yTest.OneHotEncode(10)
	yTest.Reshape([]int{yTest.Shape().Rows(), 1, 1, yTest.Shape().Cols()})

	loadedModel, err := m.LoadModel("/home/tskraan/giraffe/saved_models/mnist_0-9926_percent.json")
	if err != nil {
		return err
	}

	loadedModel.History()

	for i := range xTest.Shape().Batches()-1 {
		testCase, _ := xTest.BatchSlice(i, i+1)
		yTrue, _ := yTest.BatchSlice(i, i+1)

		test, err := loadedModel.Evaluate(testCase)
		if err != nil {
			fmt.Println("evaluate messed up")
			return err
		}

		Predicted, err := test.ArgMax(1)
		True, err := yTrue.ArgMax(1)

		fmt.Printf("Predicted Value: %v\nTrue Value: %v\n", Predicted, True)
	}

	return nil
}

func MNISTConvolutionalNeuralNetwork() error {
	xTrainAll, yTrainAll, err := u.LoadCSV("/home/tskraan/giraffe/data/mnist/mnist_train.csv")
	if err != nil {
		return err
	}

	t.Shuffle(xTrainAll, yTrainAll)

	shape := []int{28, 28}
	xTrainAll.Reshape(shape)
	xTrainAll.ScalarDivide(255.0, true)
	yTrainAll, _ = yTrainAll.OneHotEncode(10)
	yTrainAll.Reshape([]int{yTrainAll.Shape().Rows(), 0, 1, yTrainAll.Shape().Cols()})

	xTrain, _ := xTrainAll.BatchSlice(0, 2000)
	yTrain, _ := yTrainAll.BatchSlice(0, 2000)

	fmt.Printf("xTrain.Shape(): %v\n", xTrain.Shape())
	fmt.Printf("yTrain.Shape(): %v\n", yTrain.Shape())

	model := m.Sequential()
	model.Add(&l.Input{Shape: shape})
	model.Add(&l.Conv2D{Filters: 32, KernelSize: [2]int{2, 2}, Strides: [2]int{1, 1}, Mode: l.Full, Activation: &a.Relu{}})
	model.Add(&l.Conv2D{Filters: 64, KernelSize: [2]int{2, 2}, Strides: [2]int{1, 1}, Mode: l.Full, Activation: &a.Relu{}})
	model.Add(&l.Pooling{PoolType: l.MaxPooling, KernelSize: [2]int{2, 2}, Strides: [2]int{2, 2}, Mode: l.Full})
	model.Add(&l.Conv2D{Filters: 128, KernelSize: [2]int{2, 2}, Strides: [2]int{1, 1}, Mode: l.Full, Activation: &a.Relu{}})
	model.Add(&l.Pooling{PoolType: l.MaxPooling, KernelSize: [2]int{2, 2}, Strides: [2]int{2, 2}, Mode: l.Full})
	model.Add(&l.Flatten{})
	model.Add(&l.Dense{Units: 128, Activation: &a.Relu{}})
	model.Add(&l.Dense{Units: 10, Activation: &a.Softmax{}})

	loss := &lo.CategoricalCrossentropy{}
	optimizer := &o.Adam{}
	metrics := []string{"accuracy", "loss"}

	err = model.Compile(shape, loss, optimizer, metrics, true)
	if err != nil {
		fmt.Printf("err compiling model: %v\n", err)
		return err
	}

	err = model.Fit(xTrain, yTrain, 32, 20, true)
	if err != nil {
		fmt.Println("fitting messed up")
		return err
	}

	model.History()

	// Save the model
	err = model.SaveModel("/home/tskraan/giraffe/saved_models/saved_model.json")
	if err != nil {
		return err
	}
	// Load the model
	loadedModel, err := m.LoadModel("/home/tskraan/giraffe/saved_models/saved_model.json")
	if err != nil {
		return err
	}

	loadedModel.History()

	testCase, _ := xTrainAll.BatchSlice(5000, 5001)
	testCorrect, _ := yTrainAll.BatchSlice(5000, 5001)

	test, err := loadedModel.Evaluate(testCase)
	if err != nil {
		fmt.Println("evaluate messed up")
		return err
	}

	Predicted, err := test.ArgMax(1)
	True, err := testCorrect.ArgMax(1)

	fmt.Printf("Predicted Value: %v\nTrue Value: %v\n", Predicted, True)
	return nil
}

func testDenseBreastcancer() error {

	xTrain, yTrain, err := u.LoadCSV("/home/tskraan/giraffe/data/breastcancer/data.csv")
	xTrain.Normalize()
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return err
	}

  fmt.Printf("xTrain.Shape(): %v\n", xTrain.Shape())
	columns := xTrain.Shape().Cols()
	shape := []int{1, columns}
	err = xTrain.Reshape(shape)
	if err != nil {
		return err
	}
	yTrain.Reshape([]int{yTrain.Shape().Rows(), 1, 1, 1})

	model := m.Sequential()
	model.Add(&l.Input{Shape: shape})
	model.Add(&l.Dense{Units: 512, Activation: &a.Relu{}})
	model.Add(&l.Dense{Units: 256, Activation: &a.Relu{}})
	model.Add(&l.Dense{Units: 128, Activation: &a.Relu{}})
	model.Add(&l.Dense{Units: 1, Activation: &a.Sigmoid{}})

	loss := &lo.BinaryCrossEntropy{}
	optimizer := o.Adam{}

	metrics := []string{"accuracy", "loss"}
	err = model.Compile(shape, loss, &optimizer, metrics, true)
	if err != nil {
		fmt.Printf("err compiling model: %v\n", err)
		return err
	}

	err = model.Fit(xTrain, yTrain, 32, 50, true)
	if err != nil {
		fmt.Println("fitting messed up")
		return err
	}

  fmt.Println()
	model.History()
  fmt.Println()

	// should be 1
	testCase, err := xTrain.BatchSlice(500, 501)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return err
	}

	testCorrect, err := yTrain.Slice(500, 501)
	if err != nil {
		return err
	}

	// Save the model
	err = model.SaveModel("/home/tskraan/giraffe/saved_models/saved_model.json")
	if err != nil {
		return err
	}
	// Load the model
	loadedModel, err := m.LoadModel("/home/tskraan/giraffe/saved_models/saved_model.json")
	if err != nil {
		return err
	}

	loadedModel.History()

	test, err := loadedModel.Evaluate(testCase)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		fmt.Println("evaluate messed up")
		return err
	}

	fmt.Printf("Predicted answer: %v\n", test)
	fmt.Printf("Correct answer: %v\n", testCorrect)

	return nil
}
