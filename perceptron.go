package main

func And(x1, x2 float64) bool {
	w1, w2, t := 0.5, 0.5, 0.7 // weights and threshold
	return x1*w1+x2*w2 > t
}
