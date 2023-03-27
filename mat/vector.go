package mat

import "gonum.org/v1/gonum/mat"

type Vector interface {
	mat.Vector
}

func NewVector(v []float64) Vector {
	return mat.NewVecDense(len(v), v)
}

func Dot(a, b Vector) float64 {
	return mat.Dot(a, b)
}
