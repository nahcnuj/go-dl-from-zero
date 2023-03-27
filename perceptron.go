package main

import "gonum.org/v1/gonum/mat"

func And(x mat.Vector) bool {
	w := mat.NewVecDense(2, []float64{0.5, 0.5}) // weight
	b := -0.7                                    // bias
	return mat.Dot(x, w)+b > 0
}
