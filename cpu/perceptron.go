package cpu

import "gonum.org/v1/gonum/mat"

func And(x mat.Vector) bool {
	w := mat.NewVecDense(2, []float64{0.5, 0.5})
	b := -0.7
	return mat.Dot(x, w)+b > 0
}
