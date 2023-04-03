package chap3

import "github.com/nahcnuj/go-dl-from-zero/calculator"

func Step[T calculator.Scalar](be calculator.Backend[T], v calculator.Vector[T]) (calculator.Vector[T], error) {
	return be.VectorElementWiseGreaterThan(v, be.ZeroVector(v.Dim()))
}

func Sigmoid[T calculator.Scalar](be calculator.Backend[T], v calculator.Vector[T]) (calculator.Vector[T], error) {
	return be.Sigmoid(v)
}

func ReLU[T calculator.Scalar](be calculator.Backend[T], v calculator.Vector[T]) (calculator.Vector[T], error) {
	return be.ReLU(v)
}
