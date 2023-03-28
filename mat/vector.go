package mat

import (
	"gonum.org/v1/gonum/mat"
)

type cpuVector struct {
	*mat.VecDense
}

// NewVector は引数の要素を持つベクトルを作成する。
func (*CPUBackend) NewVector(v []float64) Vector {
	return cpuVector{mat.NewVecDense(len(v), v)}
}

func (*CPUBackend) Dot(a, b Vector) float64 {
	return mat.Dot(a.(cpuVector), b.(cpuVector))
}

func (cpu *CPUBackend) AddVectors(a, b Vector) Vector {
	ret := mat.NewVecDense(a.Len(), nil)
	ret.AddVec(a.(cpuVector), b.(cpuVector))
	return cpuVector{ret}
}
