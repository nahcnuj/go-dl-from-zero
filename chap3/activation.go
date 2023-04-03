package chap3

import "github.com/nahcnuj/go-dl-from-zero/calculator"

func Step[T calculator.Scalar](be calculator.Backend[T], v calculator.Vector[T]) (calculator.Vector[T], error) {
	return be.VectorElementWiseGreaterThan(v, be.ZeroVector(v.Dim()))
}