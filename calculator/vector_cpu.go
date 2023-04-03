package calculator

import "gonum.org/v1/gonum/mat"

// CPUVector represents a vector value on CPU, wrapping *mat.VecDense of gonum/mat.
type CPUVector struct {
	dense *mat.VecDense
}

// Dim returns the vector dimension.
func (v *CPUVector) Dim() int {
	return v.dense.Len()
}

// Raw returns a slice of elements of the vector
func (v *CPUVector) Raw() []float64 {
	return v.dense.RawVector().Data
}

// NewVector creates a vector.
func (CPUBackend) NewVector(elems []float64) Vector[float64] {
	return &CPUVector{mat.NewVecDense(len(elems), elems)}
}

// ZeroVector returns zero vector of given dimensions.
func (cpu CPUBackend) ZeroVector(dim int) Vector[float64] {
	return cpu.NewVector(make([]float64, dim))
}

// AddVectors adds given vectors and returns the result.
func (CPUBackend) AddVectors(vs ...Vector[float64]) (Vector[float64], error) {
	sum := mat.NewVecDense(vs[0].Dim(), nil)
	for _, v := range vs {
		sum.AddVec(sum, v.(*CPUVector).dense)
	}
	return &CPUVector{sum}, nil
}

// Dot returns the dot product of given two vectors.
func (CPUBackend) Dot(x, y Vector[float64]) (float64, error) {
	return mat.Dot(x.(*CPUVector).dense, y.(*CPUVector).dense), nil
}

// VectorElementWiseGreaterThan compares each element of given two vectors, returns a vector that each element is 1 if
func (cpu CPUBackend) VectorElementWiseGreaterThan(x, y Vector[float64]) (Vector[float64], error) {
	xs, ys := x.Raw(), y.Raw()

	ret := make([]float64, x.Dim())
	for i := range x.Raw() {
		if xs[i] > ys[i] {
			ret[i] = 1
		}
	}
	return cpu.NewVector(ret), nil
}
