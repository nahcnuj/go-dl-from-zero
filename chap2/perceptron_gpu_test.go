//go:build gpu

package chap2

import (
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
)

func TestAndByGPU(t *testing.T) {
	gpu, err := calculator.NewGPUBackend()
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Release()

	testAnd(t, gpu)
}

func TestNandByGPU(t *testing.T) {
	gpu, err := calculator.NewGPUBackend()
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Release()

	testNand(t, gpu)
}

func TestOrByGPU(t *testing.T) {
	gpu, err := calculator.NewGPUBackend()
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Release()

	testOr(t, gpu)
}

func TestXorByGPU(t *testing.T) {
	gpu, err := calculator.NewGPUBackend()
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Release()

	testXor(t, gpu)
}
