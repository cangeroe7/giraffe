package optimizers

import (
	"errors"
	"fmt"
	"math"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type Adam struct {
	LearningRate float64             // Defaults to 0.001
	Beta1        float64             // Defaults to 0.999
	Beta2        float64             // Defaults to 0.9
	Epsilon      float64             // Defaults to 1e-07
	MT           map[string]t.Tensor // First moment
	VT           map[string]t.Tensor // Second moment
	T            int                 // Time step
}

func (a *Adam) Apply(key string, param, gradient t.Tensor) error {
	if param == nil || gradient == nil {
		return nil
	}

	a.T++

	// For initialization
	if _, ok := a.MT[key]; !ok {
		shape := param.Shape()

		a.MT[key] = t.ZerosTensor(shape)
		a.VT[key] = t.ZerosTensor(shape)
	}

	mt := a.MT[key]
	vt := a.VT[key]

	// Update first moment
	updateMT := func(vals ...float64) (float64, error) {
		if len(vals) != 2 {
			return 0.0, errors.New("Must be 2 values")
		}

		mt, grad := vals[0], vals[1]
		return a.Beta1*mt + (1-a.Beta1)*grad, nil
	}

	newMT, err := mt.MapBatch(updateMT, true, gradient)
	if err != nil {
		fmt.Printf("err updating mt: %v\n", err)
		return err
	}

	//Update second moment
	updateVT := func(vals ...float64) (float64, error) {
		if len(vals) != 2 {
			return 0.0, errors.New("Must be 2 values")
		}

		vt, grad := vals[0], vals[1]
		return a.Beta1*vt + (1-a.Beta1)*grad*grad, nil
	}

	newVT, err := vt.MapBatch(updateVT, true, gradient)
	if err != nil {
		fmt.Printf("err updating VT: %v\n", err)
		return err
	}

	// Bias correction for initialization first moment
	getMTHat := func(mt float64) (float64, error) {
		return mt / (1 - math.Pow(a.Beta1, float64(a.T))), nil
	}

	mtHat, err := newMT.Map(getMTHat, false)
	if err != nil {
		fmt.Printf("err getting mthat: %v\n", err)
		return err
	}

	// Bias correction for initialization second moment
	getVTHat := func(vt float64) (float64, error) {
		return vt / (1 - math.Pow(a.Beta2, float64(a.T))), nil
	}

	vtHat, err := newVT.Map(getVTHat, false)
	if err != nil {
		fmt.Printf("err getting vthat: %v\n", err)
		return err
	}

	// Calculate the scaled and bias-corrected gradient
	scaleGrad := func(vals ...float64) (float64, error) {
		if len(vals) != 2 {
			return 0.0, errors.New("Must be 2 values")
		}

		mHat, vHat := vals[0], vals[1]
		return a.LearningRate * mHat / (math.Sqrt(vHat) + a.Epsilon), nil
	}

	scaledGrad, err := mtHat.MapBatch(scaleGrad, false, vtHat)
	if err != nil {
		fmt.Printf("err scaling grad: %v\n", err)
		return err
	}

	// Update the parameter
	_, err = param.Subtract(scaledGrad, true)
	if err != nil {
		fmt.Printf("err updating params: %v\n", err)
		return err
	}

	return nil
}

func (a *Adam) Initialize() error {
	if a.LearningRate < 0.0 {
		return errors.New("Learning rate cannot be negative")
	}
	if a.LearningRate == 0.0 {
		a.LearningRate = 0.001
	}

	if a.Beta1 < 0.0 {
		return errors.New("Beta 1 cannot be negative")
	}
	if a.Beta1 == 0.0 {
		a.Beta1 = 0.9
	}

	if a.Beta2 < 0.0 {
		return errors.New("Beta 2 cannot be negative")
	}
	if a.Beta2 == 0.0 {
		a.Beta2 = 0.999
	}

	if a.Epsilon < 0.0 {
		return errors.New("Beta 1 cannot be negative")
	}
	if a.Epsilon == 0.0 {
		a.Epsilon = 1e-07
	}

	if a.MT == nil {
		a.MT = make(map[string]t.Tensor)
	}

	if a.VT == nil {
		a.VT = make(map[string]t.Tensor)
	}

	return nil
}
