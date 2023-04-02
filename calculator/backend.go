package calculator

type Scalar interface {
	float32 | float64
}

type Vector[T Scalar] interface {
	Dim() int

	Raw() []T
}

type Backend[T Scalar] interface {
	Release()

	NewVector(elems []T) Vector[T]
	AddVectors(vs ...Vector[T]) (Vector[T], error)
	Dot(x, y Vector[T]) (T, error)
}