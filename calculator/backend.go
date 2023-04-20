package calculator

type Scalar interface {
	float32 | float64
}

type Vector[T Scalar] interface {
	Dim() int

	Raw() []T
}

type Matrix[T Scalar] interface {
	Rows() int
	Cols() int

	Raw() []T
	RawRow(r int) []T
	RawCol(c int) []T
}

type Backend[T Scalar] interface {
	Release()

	NewVector(elems []T) Vector[T]
	ZeroVector(dim int) Vector[T]
	AddVectors(vs ...Vector[T]) (Vector[T], error)
	Dot(x, y Vector[T]) (T, error)
	VectorElementWiseGreaterThan(x, y Vector[T]) (Vector[T], error)
	Sigmoid(x Vector[T]) (Vector[T], error)
	ReLU(x Vector[T]) (Vector[T], error)

	NewMatrix(elems [][]T) Matrix[T]
	Multiply(x, y Matrix[T]) (Matrix[T], error)
}
