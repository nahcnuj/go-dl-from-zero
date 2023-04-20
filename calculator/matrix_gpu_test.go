//go:build gpu

package calculator_test

import (
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
)

func TestMultiply(t *testing.T) {
	gpu, err := calculator.NewGPUBackend()
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Release()

	testMultiply(t, gpu)
}
