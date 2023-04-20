//go:build cpu

package calculator_test

import (
	"os"
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
)

var cpu calculator.Backend[float64]

func TestMain(m *testing.M) {
	cpu = calculator.NewCPUBackend()

	ret := m.Run()
	os.Exit(ret)
}
