package layers

import (
	"errors"

	t "github.com/cangeroe7/giraffe/pgk/tensor"
)

type PoolingType string

const (
	MaxPooling PoolingType = "max"
	MinPooling PoolingType = "min"
	AvgPooling PoolingType = "avg"
)

type Pooling struct {
	PoolType   PoolingType
	KernelSize [2]int
	Strides    [2]int
	Mode       PaddingMode

  padding []int

  outShape t.Shape
	input t.Tensor
}

func (p *Pooling) CompileLayer(inShape t.Shape) (t.Shape, error) {

	// Set padding values
	padding, err := ComputePadding(inShape, p.KernelSize, p.Strides, p.Mode)
  if err != nil {
    return nil, err
  }

  p.padding = padding

	// Compute output shape
	outHeight := (p.padding[0]+p.padding[2]+inShape.Rows()-p.KernelSize[0])/p.Strides[0] + 1
	outWidth := (p.padding[1]+p.padding[3]+inShape.Cols()-p.KernelSize[0])/p.Strides[1] + 1

	var outShape t.Shape = []int{p.input.Shape().Channels(), outHeight, outWidth}

	return outShape, nil
}

func (p *Pooling) Forward(input t.Tensor) (t.Tensor, error) {
	if input == nil {
		return nil, errors.New("input cannot be nil")
	}

	// Store the input for the backward pass
	p.input = input

	// Calculate output dimensions
	inputShape := input.Shape().Clone()
	outHeight := (inputShape.Rows()-p.KernelSize[0])/p.Strides[0] + 1
	outWidth := (inputShape.Cols()-p.KernelSize[1])/p.Strides[1] + 1
	outShape := []int{inputShape.Batches(), inputShape.Channels(), outHeight, outWidth}
	output := t.ZerosTensor(outShape)

	// Iterate through batches and channels
	channelIter, _ := t.IterFromTensor(input, "channel")
	outIter, _ := t.IterFromTensor(output, "channel")

	for inChannel, ok := channelIter.Next(); ok; inChannel, ok = channelIter.Next() {
		outChannel, _ := outIter.Next()

		// Apply pooling operation
		for i := 0; i < outHeight; i++ {
			for j := 0; j < outWidth; j++ {
				region, err := inChannel.RegionSlice(p.Strides[0]*i, p.Strides[1]*j, p.Strides[0], p.Strides[0])
				if err != nil {
					return nil, err
				}

				switch p.PoolType {
				case MaxPooling:
					outChannel.SetValueAt(i*outWidth+j, region.Max())
				case MinPooling:
					outChannel.SetValueAt(i*outWidth+j, region.Min())
				case AvgPooling:
					outChannel.SetValueAt(i*outWidth+j, region.Avg())
				}
			}
		}
	}

	return output, nil
}

func (p *Pooling) Backward(gradient t.Tensor) (t.Tensor, error) {
	if gradient == nil {
		return nil, errors.New("gradient tensor cannot be nil")
	}

	if gradient.Shape().Rows() != (p.input.Shape().Rows()-p.KernelSize[0])/p.Strides[0]+1 ||
		gradient.Shape().Cols() != (p.input.Shape().Cols()-p.KernelSize[1])/p.Strides[1]+1 {
		return nil, errors.New("gradient shape does not match output shape of forward pass")
	}

	outputGradient := t.ZerosTensor(p.input.Shape().Clone())

	channelIter, _ := t.IterFromTensor(p.input, "channel")
	gradientIter, _ := t.IterFromTensor(gradient, "channel")
	outGradientIter, _ := t.IterFromTensor(outputGradient, "channel")

	for inChannel, ok := channelIter.Next(); ok; inChannel, ok = channelIter.Next() {
		gradientChannel, _ := gradientIter.Next()
		outChannel, _ := outGradientIter.Next()

		for i := 0; i < gradient.Shape().Rows(); i++ {
			for j := 0; j < gradient.Shape().Cols(); j++ {

				startRow := p.Strides[0] * i
				startCol := p.Strides[1] * j

				// Slice the region from the input
				region, err := inChannel.RegionSlice(startRow, startCol, p.KernelSize[0], p.KernelSize[1])
				if err != nil {
					return nil, err
				}

				switch p.PoolType {
				case MaxPooling:
					// Get local max index
					localMaxIdx := region.MaxIndex()

					// Convert local index to global index
					localRow := localMaxIdx / p.KernelSize[1]
					localCol := localMaxIdx % p.KernelSize[1]

					globalRow := startRow + localRow
					globalCol := startCol + localCol

					// Update gradient in the output channel
					outChannel.SetValueAt(globalRow*p.input.Shape().Cols()+globalCol,
						outChannel.ValueAt(globalRow*p.input.Shape().Cols()+globalCol)+
							gradientChannel.ValueAt(i*gradient.Shape().Cols()+j))

				case MinPooling:
					// Get local min index
					localMinIdx := region.MinIndex()

					// Convert local index to global index
					localRow := localMinIdx / p.KernelSize[1]
					localCol := localMinIdx % p.KernelSize[1]
					globalRow := startRow + localRow
					globalCol := startCol + localCol

					// Update gradient in the output channel
					outChannel.SetValueAt(globalRow*p.input.Shape().Cols()+globalCol,
						outChannel.ValueAt(globalRow*p.input.Shape().Cols()+globalCol)+
							gradientChannel.ValueAt(i*gradient.Shape().Cols()+j))

				case AvgPooling:
					// Distribute the gradient equally across the region
					avgGradient := gradientChannel.ValueAt(i*gradient.Shape().Cols()+j) / float64(region.Size())
					for localIdx := 0; localIdx < region.Size(); localIdx++ {
						localRow := localIdx / p.KernelSize[1]
						localCol := localIdx % p.KernelSize[1]
						globalRow := startRow + localRow
						globalCol := startCol + localCol

						// Update gradient in the output channel
						outChannel.SetValueAt(globalRow*p.input.Shape().Cols()+globalCol,
							outChannel.ValueAt(globalRow*p.input.Shape().Cols()+globalCol)+avgGradient)
					}
				}
			}
		}
	}

	return outputGradient, nil
}

func (p *Pooling) Type() string {
	return string(p.PoolType) + "pooling"
}

func (p *Pooling) Weights() t.Tensor         { return nil }
func (p *Pooling) Biases() t.Tensor          { return nil }
func (p *Pooling) WeightsGradient() t.Tensor { return nil }
func (p *Pooling) BiasesGradient() t.Tensor  { return nil }
