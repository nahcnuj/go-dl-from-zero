package chap2

import "github.com/nahcnuj/go-dl-from-zero/calculator"

// And calculates AND of given two numbers by perceptron mechanism.
func And[T calculator.Scalar](be calculator.Backend[T], x1, x2 T) (bool, error) {
	var (
		x   = be.NewVector([]T{x1, x2})
		w   = be.NewVector([]T{0.5, 0.5})
		b T = -0.7
	)
	dot, err := be.Dot(x, w)
	if err != nil {
		return false, err
	}
	return dot+b > 0, nil
}

// Nand calculates NAND of given two numbers by perceptron mechanism.
func Nand[T calculator.Scalar](be calculator.Backend[T], x1, x2 T) (bool, error) {
	var (
		x   = be.NewVector([]T{x1, x2})
		w   = be.NewVector([]T{-0.5, -0.5})
		b T = 0.7
	)
	dot, err := be.Dot(x, w)
	if err != nil {
		return false, err
	}
	return dot+b > 0, nil
}

// Or calculates OR of given two numbers by perceptron mechanism.
func Or[T calculator.Scalar](be calculator.Backend[T], x1, x2 T) (bool, error) {
	var (
		x   = be.NewVector([]T{x1, x2})
		w   = be.NewVector([]T{0.5, 0.5})
		b T = -0.2
	)
	dot, err := be.Dot(x, w)
	if err != nil {
		return false, err
	}
	return dot+b > 0, nil
}
