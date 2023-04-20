package calculator

import "gonum.org/v1/gonum/mat"

// CPUMatrix represents a matrix value for computation on CPU
type CPUMatrix struct {
	dense *mat.Dense
}

// Rows returns the number of rows of the matrix.
func (m *CPUMatrix) Rows() int {
	return m.dense.RawMatrix().Rows
}

// Cols returns the number of columns of the matrix.
func (m *CPUMatrix) Cols() int {
	return m.dense.RawMatrix().Cols
}

// Raw returns an one-dimensional slice which packed elements of the matrix in.
func (m *CPUMatrix) Raw() []float64 {
	return m.dense.RawMatrix().Data
}

// RawRow returns the values of `r`-th row of the matrix.
func (m *CPUMatrix) RawRow(r int) []float64 {
	return m.dense.RawRowView(r)
}

// RawCol returns the values of `c`-th column of the matrix.
func (m *CPUMatrix) RawCol(c int) []float64 {
	return m.dense.T().(*mat.Dense).RawRowView(c)
}

// NewMatrix creates a matrix.
func (CPUBackend) NewMatrix(elems [][]float64) Matrix[float64] {
	r, c := len(elems), len(elems[0])
	data := make([]float64, 0, r*c)
	for _, cs := range elems {
		data = append(data, cs...)
	}
	return &CPUMatrix{dense: mat.NewDense(r, c, data)}
}

// Multiply multiplies given two matrices and returns the result.
func (cpu CPUBackend) Multiply(x, y Matrix[float64]) (Matrix[float64], error) {
	r, c := x.Rows(), y.Cols()

	mul := mat.NewDense(r, c, nil)
	mul.Mul(x.(*CPUMatrix).dense, y.(*CPUMatrix).dense)

	elems := make([][]float64, r)
	for i := range elems {
		elems[i] = mul.RawRowView(i)
	}
	return cpu.NewMatrix(elems), nil
}
