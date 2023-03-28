package main

import "go-dl-from-zero/mat"

func (dl *DeepLearning) And(x mat.Vector) bool {
	w := dl.NewVector([]float64{0.5, 0.5})
	b := -0.7
	return dl.Dot(x, w)+b > 0
}
