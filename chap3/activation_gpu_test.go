//go:build gpu

package chap3

import (
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
)

func TestStepByGPU(t *testing.T) {
	gpu, err := calculator.NewGPUBackend()
	if err != nil {
		t.Fatal(err)
	}

	testStep(t, gpu)
}
