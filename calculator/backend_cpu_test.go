//go:build cpu

package calculator_test

import (
	"os"
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
)

const (
	delta   = 1e-5 // 許容絶対誤差
	epsilon = 0.1  // 許容相対誤差
)

var cpu calculator.Backend[float64]

func TestMain(m *testing.M) {
	cpu = calculator.NewCPUBackend()

	ret := m.Run()
	os.Exit(ret)
}
