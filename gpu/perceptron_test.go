//go:build gpu

package gpu_test

import (
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/gpu"
)

func TestAnd(t *testing.T) {
	gpu, err := gpu.NewBackend()
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Release()

	tests := []struct {
		name string
		x    [2]float32
		want bool
	}{
		{name: "0 and 0 == false", x: [2]float32{0, 0}, want: false},
		{name: "1 and 0 == false", x: [2]float32{1, 0}, want: false},
		{name: "0 and 1 == false", x: [2]float32{0, 1}, want: false},
		{name: "1 and 1 == false", x: [2]float32{1, 1}, want: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			x := gpu.NewVecDense(tc.x[:])
			got, err := gpu.And(x)
			if err != nil {
				t.Fatal(err)
			}

			if got != tc.want {
				t.Errorf("expected: %v, got: %v", tc.want, got)
			}
		})
	}
}
