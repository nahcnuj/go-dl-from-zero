//go:build gpu

package gmat

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

type Vector interface {
	Len() int
	AtVec(i int) float32
}

type VecDense struct {
	Data []float32
}

func (*Backend) NewVecDense(v []float32) *VecDense {
	return &VecDense{v}
}

func (*Backend) ZeroVector(dim int) Vector {
	return &VecDense{make([]float32, dim)}
}

func (v *VecDense) Len() int {
	return len(v.Data)
}

func (v *VecDense) AtVec(i int) float32 {
	return v.Data[i]
}

func (b *Backend) AddVectors(x Vector, y Vector) (add Vector, err error) {
	k, ok := b.kernels["vec_add"]
	if !ok {
		err = errors.New("kernel function not found")
		return
	}

	dim := uint64(x.Len())

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

	if err = queue.EnqueueWriteBuffer(aBuf, true, x.(*VecDense).Data); err != nil {
		return
	}
	if err = queue.EnqueueWriteBuffer(bBuf, true, y.(*VecDense).Data); err != nil {
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
	add = &VecDense{ret}
	return
}

func (b *Backend) Dot(x Vector, y Vector) (dot float32, err error) {
	k, ok := b.kernels["vec_dot"]
	if !ok {
		err = errors.New("kernel function not found")
		return
	}

	dim := uint64(x.Len())

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

	if err = queue.EnqueueWriteBuffer(aBuf, true, x.(*VecDense).Data); err != nil {
		return
	}
	if err = queue.EnqueueWriteBuffer(bBuf, true, y.(*VecDense).Data); err != nil {
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
