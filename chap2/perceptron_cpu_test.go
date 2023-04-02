package chap2

import (
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
)

func TestAndByCPU(t *testing.T) {
	cpu := calculator.NewCPUBackend()

	testAnd(t, cpu)
}

func TestNandByCPU(t *testing.T) {
	cpu := calculator.NewCPUBackend()

	testNand(t, cpu)
}

func TestOrByCPU(t *testing.T) {
	cpu := calculator.NewCPUBackend()

	testOr(t, cpu)
}

func TestXorByCPU(t *testing.T) {
	cpu := calculator.NewCPUBackend()

	testXor(t, cpu)
}
