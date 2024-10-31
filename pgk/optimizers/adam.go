package optimizers

import (
	"errors"
	"fmt"
	"math"

	u "github.com/cangeroe7/giraffe/internal/utils"
)

type Adam struct {
	LearningRate float64             // Defaults to 0.001
	Beta1        float64             // Defaults to 0.999
	Beta2        float64             // Defaults to 0.9
	Epsilon      float64             // Defaults to 1e-07
	MT           map[string]u.Matrix // First moment
	VT           map[string]u.Matrix // Second moment
	T            int                 // Time step
}

func (a *Adam) Apply(key string, param, gradient u.Matrix) error {
  if param == nil || gradient == nil {
    return nil
  }

	a.T++

	// For initialization
	if _, ok := a.MT[key]; !ok {
		shape := param.Shape()

		a.MT[key] = u.ZerosMatrix(shape[0], shape[1])
		a.VT[key] = u.ZerosMatrix(shape[0], shape[1])
	}

	mt := a.MT[key]
	vt := a.VT[key]

	// Update first moment
	updateMT := func(mt, grad float64) (float64, error) {
		return a.Beta1*mt + (1-a.Beta1)*grad, nil
	}

	newMT, err := mt.MapOnto(updateMT, gradient, true)
	if err != nil {
    fmt.Printf("err updating mt: %v\n", err)
		return err
	}

	//Update second moment
	updateVT := func(vt, grad float64) (float64, error) {
		return a.Beta1*vt + (1-a.Beta1)*grad*grad, nil
	}

	newVT, err := vt.MapOnto(updateVT, gradient, true)
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
	scaleGrad := func(mHat, vHat float64) (float64, error) {
		return a.LearningRate * mHat / (math.Sqrt(vHat) + a.Epsilon), nil
	}

	scaledGrad, err := mtHat.MapOnto(scaleGrad, vtHat, false)
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
    a.MT = make(map[string]u.Matrix)
  }

  if a.VT == nil {
    a.VT = make(map[string]u.Matrix)
  }

	return nil
}
