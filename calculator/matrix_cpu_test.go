//go:build cpu

package calculator_test

import (
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
)

func TestMultiply(t *testing.T) {
	cpu := calculator.NewCPUBackend()

	testMultiply(t, cpu)
}
