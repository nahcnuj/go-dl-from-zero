//go:build gpu

package calculator_test

import (
	"os"
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
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
