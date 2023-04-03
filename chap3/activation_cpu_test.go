package chap3

import (
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
)

func TestStepByCPU(t *testing.T) {
	cpu := calculator.NewCPUBackend()

	testStep(t, cpu)
}
