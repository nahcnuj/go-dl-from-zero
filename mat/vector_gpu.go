//go:build gpu

package mat

import (
	"unsafe"

	"github.com/PassKeyRa/go-opencl/opencl"
)

var (
	dummy = struct {
		float float32
	}{}
	floatSize = uint64(unsafe.Sizeof(dummy.float))
)

type gpuVector struct {
	N    int
	Data []float32
}

func (v *gpuVector) Len() int {
	return v.N
}

func (v *gpuVector) AtVec(i int) float64 {
	return float64(v.Data[i])
}

// 注意：引数の型は float64 のスライスであるが、OpenCL の都合で計算は float32 で行われる。
func (gpu *GPUBackend) NewVector(v []float64) Vector {
	downsized := make([]float32, len(v))
	for i, v := range v {
		downsized[i] = float32(v)
	}
	return &gpuVector{N: len(v), Data: downsized}
}

func (gpu *GPUBackend) ZeroVector(dim int) Vector {
	return newZeroVector(gpu, dim)
}

func (gpu *GPUBackend) AddVectors(a Vector, b Vector) Vector {
	return gpu.addVectors(a.(*gpuVector), b.(*gpuVector))
}

func (gpu *GPUBackend) addVectors(a *gpuVector, b *gpuVector) *gpuVector {
	add, ok := gpu.device.kernels["vec_add"]
	if !ok {
		panic("kernel function not found")
	}

	dim := uint64(a.N)

	retBuf, err := gpu.device.context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, dim*floatSize)
	if err != nil {
		panic(err)
	}
	defer retBuf.Release()

	aBuf, err := gpu.device.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dim*floatSize)
	if err != nil {
		panic(err)
	}
	defer aBuf.Release()

	bBuf, err := gpu.device.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dim*floatSize)
	if err != nil {
		panic(err)
	}
	defer bBuf.Release()

	if err = add.SetArg(0, retBuf.Size(), &retBuf); err != nil {
		panic(err)
	}
	if err = add.SetArg(1, aBuf.Size(), &aBuf); err != nil {
		panic(err)
	}
	if err = add.SetArg(2, bBuf.Size(), &bBuf); err != nil {
		panic(err)
	}

	wa, wb := make([]float32, dim), make([]float32, dim)
	for i := 0; i < a.N; i++ {
		wa[i] = a.Data[i]
		wb[i] = b.Data[i]
	}
	gpu.device.queue.EnqueueWriteBuffer(aBuf, true, wa)
	gpu.device.queue.EnqueueWriteBuffer(bBuf, true, wb)

	if err = gpu.device.queue.EnqueueNDRangeKernel(add, 1, []uint64{128}); err != nil {
		panic(err)
	}

	gpu.device.queue.Flush()
	gpu.device.queue.Finish()

	ret := make([]float32, dim)
	if err = gpu.device.queue.EnqueueReadBuffer(retBuf, true, ret); err != nil {
		panic(err)
	}

	return &gpuVector{N: len(ret), Data: ret}
}
