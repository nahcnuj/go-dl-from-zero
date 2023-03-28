package mat

import (
	"gonum.org/v1/gonum/mat"
)

type Vector interface {
	Len() int
	AtVec(int) float64
}

type cpuVector struct {
	*mat.VecDense
}

func (*CPUBackend) NewVector(v []float64) Vector {
	return cpuVector{mat.NewVecDense(len(v), v)}
}

func (cpu *CPUBackend) ZeroVector(dim int) Vector {
	return newZeroVector(cpu, dim)
}

func (*CPUBackend) Dot(a, b Vector) float64 {
	return mat.Dot(a.(cpuVector), b.(cpuVector))
}

func (cpu *CPUBackend) AddVectors(a, b Vector) Vector {
	ret := mat.NewVecDense(a.Len(), nil)
	ret.AddVec(a.(cpuVector), b.(cpuVector))
	return cpuVector{ret}
}
