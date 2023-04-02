package chap2

import "github.com/nahcnuj/go-dl-from-zero/calculator"

// And calculates AND of given two numbers by perceptron mechanism.
func And[T calculator.Scalar](be calculator.Backend[T], x1, x2 T) (T, error) {
	var (
		x   = be.NewVector([]T{x1, x2})
		w   = be.NewVector([]T{0.5, 0.5})
		b T = -0.7
	)
	dot, err := be.Dot(x, w)
	if err != nil {
		return 0, err
	}
	if dot+b > 0 {
		return 1, nil
	} else {
		return 0, nil
	}
}

// Nand calculates NAND of given two numbers by perceptron mechanism.
func Nand[T calculator.Scalar](be calculator.Backend[T], x1, x2 T) (T, error) {
	var (
		x   = be.NewVector([]T{x1, x2})
		w   = be.NewVector([]T{-0.5, -0.5})
		b T = 0.7
	)
	dot, err := be.Dot(x, w)
	if err != nil {
		return 0, err
	}

	if dot+b > 0 {
		return 1, nil
	} else {
		return 0, nil
	}
}

// Or calculates OR of given two numbers by perceptron mechanism.
func Or[T calculator.Scalar](be calculator.Backend[T], x1, x2 T) (T, error) {
	var (
		x   = be.NewVector([]T{x1, x2})
		w   = be.NewVector([]T{0.5, 0.5})
		b T = -0.2
	)
	dot, err := be.Dot(x, w)
	if err != nil {
		return 0, err
	}

	if dot+b > 0 {
		return 1, nil
	} else {
		return 0, nil
	}
}

// Xor calculates XOR of given two numbers by perceptron mechanism.
//
//	(x₁)-(s₁)
//	    ☓    >( y )
//	(x₂)-(s₂)
func Xor[T calculator.Scalar](be calculator.Backend[T], x1, x2 T) (T, error) {
	s1, err := Nand(be, x1, x2)
	if err != nil {
		return 0, nil
	}

	s2, err := Or(be, x1, x2)
	if err != nil {
		return 0, nil
	}

	return And(be, s1, s2)
}
