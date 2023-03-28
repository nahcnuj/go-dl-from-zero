//go:build gpu

package main

import (
	"os"
	"testing"

	"go-dl-from-zero/mat"
)

var dl DeepLearning

func TestMain(m *testing.M) {
	gpu, err := mat.NewGPUBackend()
	if err != nil {
		panic(err)
	}
	defer gpu.Release()

	dl = NewDeepLearning(gpu)

	ret := m.Run()
	os.Exit(ret)
}

func TestAndByGPU(t *testing.T) {
	tests := []struct {
		name string
		x    [2]float64
		want bool
	}{
		{name: "0 and 0 gets 0", x: [2]float64{0, 0}, want: false},
		{name: "1 and 0 gets 0", x: [2]float64{1, 0}, want: false},
		{name: "0 and 1 gets 0", x: [2]float64{0, 1}, want: false},
		{name: "1 and 1 gets 0", x: [2]float64{1, 1}, want: true},
	}

	for _, tc := range tests {
		x := dl.NewVector(tc.x[:])
		got := dl.And(x)

		if got != tc.want {
			t.Fatalf("%s: expected: %v, got: %v", tc.name, tc.want, got)
		}
	}
}
