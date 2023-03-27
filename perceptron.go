package main

func And(x [2]float64) bool {
	w := [2]float64{0.5, 0.5} // weights
	b := -0.7                 // bias
	return b+x[0]*w[0]+x[1]*w[1] > 0
}
