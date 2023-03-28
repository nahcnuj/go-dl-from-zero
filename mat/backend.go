package mat

type Backend interface {
	VectorConstructor

	ZeroVector(dim int) Vector
	AddVectors(a, b Vector) Vector

	Release()
}

type VectorConstructor interface {
	// NewVector は引数の要素を持つベクトルを作成する。
	NewVector(v []float64) Vector
}

func newZeroVector(be VectorConstructor, dim int) Vector {
	return be.NewVector(make([]float64, dim))
}

type CPUBackend struct{}

func NewCPUBackend() Backend {
	return &CPUBackend{}
}

func (*CPUBackend) Release() {
	// do nothing
}
