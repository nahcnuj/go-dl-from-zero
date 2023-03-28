package mat

type Backend interface {
	NewVector(v []float64) Vector
	AddVectors(a, b Vector) Vector

	Release()
}

type CPUBackend struct{}

func NewCPUBackend() Backend {
	return &CPUBackend{}
}

func (*CPUBackend) Release() {
	// do nothing
}

type Vector interface {
	Len() int
	AtVec(int) float64
}
