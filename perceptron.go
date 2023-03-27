package main

import "go-dl-from-zero/mat"

func And(x mat.Vector) bool {
	w := mat.NewVector([]float64{0.5, 0.5}) // weight
	b := -0.7                               // bias
	return mat.Dot(x, w)+b > 0
}
