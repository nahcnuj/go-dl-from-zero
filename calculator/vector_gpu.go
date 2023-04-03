//go:build gpu

package calculator

import (
	"errors"
	"unsafe"

	"github.com/PassKeyRa/go-opencl/opencl"
)

var (
	dummy = struct {
		float float32
	}{}
	floatSize = uint64(unsafe.Sizeof(dummy.float))
)

// GPUVector represents a vector value on GPU, implementing calculator.Vector.
type GPUVector struct {
	data []float32
}

// Dim returns the vector dimension.
func (v *GPUVector) Dim() int {
	return len(v.data)
}

// Raw returns a slice of elements of the vector.
func (v *GPUVector) Raw() []float32 {
	return v.data
}

// NewVector creates a vector.
func (*GPUBackend) NewVector(elems []float32) Vector[float32] {
	return &GPUVector{data: elems}
}

// ZeroVector returns zero vector of given dimensions.
func (b *GPUBackend) ZeroVector(dim int) Vector[float32] {
	return b.NewVector(make([]float32, dim))
}

// AddVectors adds given vectors and returns the result.
func (b *GPUBackend) AddVectors(vs ...Vector[float32]) (sum Vector[float32], err error) {
	sum = b.NewVector(make([]float32, vs[0].Dim()))
	for _, v := range vs {
		sum, err = b.addTwoVectors(sum, v)
		if err != nil {
			return
		}
	}
	return
}

// Dot returns the dot product of given two vectors.
func (b *GPUBackend) Dot(x, y Vector[float32]) (float32, error) {
	return b.dot(x, y)
}

// VectorElementWiseGreaterThan compares each element of given two vectors, returns a vector that each element is 1 if
func (b *GPUBackend) VectorElementWiseGreaterThan(x, y Vector[float32]) (Vector[float32], error) {
	return b.vectorElementWiseGreaterThan(x, y)
}

// Sigmoid applys sigmoid function to given vector.
func (b *GPUBackend) Sigmoid(x Vector[float32]) (sigmoid Vector[float32], err error) {
	return b.sigmoid(x)
}

func (b *GPUBackend) addTwoVectors(x, y Vector[float32]) (sum *GPUVector, err error) {
	k, ok := b.kernels["vectorAdd"]
	if !ok {
		err = errors.New("kernel function not found")
		return
	}

	dim := uint64(x.Dim())

	retBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer retBuf.Release()

	aBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer aBuf.Release()

	bBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer bBuf.Release()

	if err = k.SetArg(0, retBuf.Size(), &retBuf); err != nil {
		return
	}
	if err = k.SetArg(1, aBuf.Size(), &aBuf); err != nil {
		return
	}
	if err = k.SetArg(2, bBuf.Size(), &bBuf); err != nil {
		return
	}

	queue, err := b.context.CreateCommandQueue(b.device)
	if err != nil {
		return
	}
	defer queue.Release()

	if err = queue.EnqueueWriteBuffer(aBuf, true, x.Raw()); err != nil {
		return
	}
	if err = queue.EnqueueWriteBuffer(bBuf, true, y.Raw()); err != nil {
		return
	}

	if err = queue.EnqueueNDRangeKernel(k, 1, []uint64{dim}); err != nil {
		return
	}

	queue.Flush()
	queue.Finish()

	ret := make([]float32, dim)
	if err = queue.EnqueueReadBuffer(retBuf, true, ret); err != nil {
		return
	}
	sum = &GPUVector{ret}
	return
}

func (b *GPUBackend) dot(x, y Vector[float32]) (dot float32, err error) {
	k, ok := b.kernels["vectorDot"]
	if !ok {
		err = errors.New("kernel function not found")
		return
	}

	dim := uint64(x.Dim())

	retBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer retBuf.Release()

	aBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer aBuf.Release()

	bBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer bBuf.Release()

	if err = k.SetArg(0, retBuf.Size(), &retBuf); err != nil {
		return
	}
	if err = k.SetArg(1, aBuf.Size(), &aBuf); err != nil {
		return
	}
	if err = k.SetArg(2, bBuf.Size(), &bBuf); err != nil {
		return
	}

	queue, err := b.context.CreateCommandQueue(b.device)
	if err != nil {
		return
	}
	defer queue.Release()

	if err = queue.EnqueueWriteBuffer(aBuf, true, x.Raw()); err != nil {
		return
	}
	if err = queue.EnqueueWriteBuffer(bBuf, true, y.Raw()); err != nil {
		return
	}

	if err = queue.EnqueueNDRangeKernel(k, 1, []uint64{dim}); err != nil {
		return
	}

	queue.Flush()
	queue.Finish()

	ret := make([]float32, dim)
	if err = queue.EnqueueReadBuffer(retBuf, true, ret); err != nil {
		return
	}

	for _, v := range ret {
		dot += v
	}
	return
}

func (b *GPUBackend) vectorElementWiseGreaterThan(x, y Vector[float32]) (stepped *GPUVector, err error) {
	k, ok := b.kernels["vectorElementWiseGreaterThan"]
	if !ok {
		err = errors.New("kernel function not found")
		return
	}

	dim := uint64(x.Dim())

	retBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer retBuf.Release()

	aBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer aBuf.Release()

	bBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer bBuf.Release()

	if err = k.SetArg(0, retBuf.Size(), &retBuf); err != nil {
		return
	}
	if err = k.SetArg(1, aBuf.Size(), &aBuf); err != nil {
		return
	}
	if err = k.SetArg(2, bBuf.Size(), &bBuf); err != nil {
		return
	}

	queue, err := b.context.CreateCommandQueue(b.device)
	if err != nil {
		return
	}
	defer queue.Release()

	if err = queue.EnqueueWriteBuffer(aBuf, true, x.Raw()); err != nil {
		return
	}
	if err = queue.EnqueueWriteBuffer(bBuf, true, y.Raw()); err != nil {
		return
	}

	if err = queue.EnqueueNDRangeKernel(k, 1, []uint64{dim}); err != nil {
		return
	}

	queue.Flush()
	queue.Finish()

	ret := make([]float32, dim)
	if err = queue.EnqueueReadBuffer(retBuf, true, ret); err != nil {
		return
	}
	stepped = &GPUVector{ret}
	return
}

func (b *GPUBackend) sigmoid(x Vector[float32]) (sigmoid *GPUVector, err error) {
	k, ok := b.kernels["vectorSigmoid"]
	if !ok {
		err = errors.New("kernel function not found")
		return
	}

	dim := uint64(x.Dim())

	retBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer retBuf.Release()

	aBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dim*floatSize)
	if err != nil {
		return
	}
	defer aBuf.Release()

	if err = k.SetArg(0, retBuf.Size(), &retBuf); err != nil {
		return
	}
	if err = k.SetArg(1, aBuf.Size(), &aBuf); err != nil {
		return
	}

	queue, err := b.context.CreateCommandQueue(b.device)
	if err != nil {
		return
	}
	defer queue.Release()

	if err = queue.EnqueueWriteBuffer(aBuf, true, x.Raw()); err != nil {
		return
	}

	if err = queue.EnqueueNDRangeKernel(k, 1, []uint64{dim}); err != nil {
		return
	}

	queue.Flush()
	queue.Finish()

	ret := make([]float32, dim)
	if err = queue.EnqueueReadBuffer(retBuf, true, ret); err != nil {
		return
	}
	sigmoid = &GPUVector{ret}
	return
}
