//go:build gpu

package calculator

import (
	"errors"

	"github.com/PassKeyRa/go-opencl/opencl"
)

// GPUMatrix represents a matrix value for computation on GPU.
type GPUMatrix struct {
	rows int
	cols int
	data []float32
}

// Rows returns the number of rows of the matrix.
func (m *GPUMatrix) Rows() int {
	return m.rows
}

// Cols returns the number of columns of the matrix.
func (m *GPUMatrix) Cols() int {
	return m.cols
}

// Raw returns an one-dimensional slice which packed elements of the matrix in.
func (m *GPUMatrix) Raw() []float32 {
	ret := make([]float32, m.rows*m.cols)
	copy(ret, m.data)
	return ret
}

// RawRow returns the values of `r`-th row of the matrix.
func (m *GPUMatrix) RawRow(r int) []float32 {
	ret := make([]float32, m.cols)
	copy(ret, m.data[r*m.cols:(r+1)*m.cols])
	return ret
}

// RawCol returns the values of `c`-th column of the matrix.
func (m *GPUMatrix) RawCol(c int) []float32 {
	ret := make([]float32, 0, m.rows)
	for l := 0; l < len(m.data); l += m.cols {
		ret = append(ret, m.data[l+c])
	}
	return ret
}

// NewMatrix creates a matrix.
func (*GPUBackend) NewMatrix(elems [][]float32) Matrix[float32] {
	r, c := len(elems), len(elems[0]) // assume that the numbers of every rows are same and the numbers of every cols are same
	data := make([]float32, 0, r*c)
	for _, cs := range elems {
		data = append(data, cs...)
	}
	return &GPUMatrix{rows: r, cols: c, data: data}
}

// NewMatrixFromRaw creates a matrix from an one-dimensional slice that packed the elements of the matrix in.
func (*GPUBackend) NewMatrixFromRaw(r, c int, elems []float32) Matrix[float32] {
	return &GPUMatrix{rows: r, cols: c, data: elems}
}

// Multiply multiplies given two matrices and returns the result.
func (b *GPUBackend) Multiply(x, y Matrix[float32]) (mul Matrix[float32], err error) {
	return b.multiply(x, y)
}

func (b *GPUBackend) multiply(x, y Matrix[float32]) (mul Matrix[float32], err error) {
	k, ok := b.kernels["matrixMultiply"]
	if !ok {
		err = errors.New("kernel function not found")
		return
	}

	r, c := x.Rows(), y.Cols()
	m := x.Cols() // assume that x.Cols() == y.Rows()

	rConstBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, floatSize)
	if err != nil {
		return
	}
	defer rConstBuf.Release()

	mConstBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, floatSize)
	if err != nil {
		return
	}
	defer mConstBuf.Release()

	cConstBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, floatSize)
	if err != nil {
		return
	}
	defer cConstBuf.Release()

	retSize := uint64(r*c) * floatSize
	retBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, retSize)
	if err != nil {
		return
	}
	defer retBuf.Release()

	xSize := uint64(r*m) * floatSize
	xBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, xSize)
	if err != nil {
		return
	}
	defer xBuf.Release()

	ySize := uint64(m*c) * floatSize
	yBuf, err := b.context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, ySize)
	if err != nil {
		return
	}
	defer yBuf.Release()

	if err = k.SetArg(0, rConstBuf.Size(), &rConstBuf); err != nil {
		return
	}
	if err = k.SetArg(1, mConstBuf.Size(), &mConstBuf); err != nil {
		return
	}
	if err = k.SetArg(2, cConstBuf.Size(), &cConstBuf); err != nil {
		return
	}
	if err = k.SetArg(3, retBuf.Size(), &retBuf); err != nil {
		return
	}
	if err = k.SetArg(4, xBuf.Size(), &xBuf); err != nil {
		return
	}
	if err = k.SetArg(5, yBuf.Size(), &yBuf); err != nil {
		return
	}

	queue, err := b.context.CreateCommandQueue(b.device)
	if err != nil {
		return
	}
	defer queue.Release()

	if err = queue.EnqueueWriteBuffer(rConstBuf, true, []float32{float32(r)}); err != nil {
		return
	}
	if err = queue.EnqueueWriteBuffer(mConstBuf, true, []float32{float32(m)}); err != nil {
		return
	}
	if err = queue.EnqueueWriteBuffer(cConstBuf, true, []float32{float32(c)}); err != nil {
		return
	}
	if err = queue.EnqueueWriteBuffer(xBuf, true, x.Raw()); err != nil {
		return
	}
	if err = queue.EnqueueWriteBuffer(yBuf, true, y.Raw()); err != nil {
		return
	}

	if err = queue.EnqueueNDRangeKernel(k, 1, []uint64{uint64(r * c)}); err != nil {
		return
	}

	queue.Flush()
	queue.Finish()

	ret := make([]float32, r*c)
	if err = queue.EnqueueReadBuffer(retBuf, true, ret); err != nil {
		return
	}
	mul = b.NewMatrixFromRaw(r, c, ret)
	return
}
