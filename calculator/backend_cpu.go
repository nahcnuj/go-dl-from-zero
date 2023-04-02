package calculator

type CPUBackend struct{}

var _ Backend[float64] = &CPUBackend{}

func NewCPUBackend() Backend[float64] {
	return &CPUBackend{}
}

func (CPUBackend) Release() {
	// do nothing
}
