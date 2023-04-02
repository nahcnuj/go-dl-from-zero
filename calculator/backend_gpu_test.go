//go:build gpu

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

var gpu calculator.Backend[float32]

func TestMain(m *testing.M) {
	var err error
	gpu, err = calculator.NewGPUBackend()
	if err != nil {
		panic(err)
	}
	defer gpu.Release()

	ret := m.Run()
	os.Exit(ret)
}
