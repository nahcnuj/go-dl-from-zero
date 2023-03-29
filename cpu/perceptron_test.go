//go:build cpu

package cpu_test

import (
	"go-dl-from-zero/cpu"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestAnd(t *testing.T) {
	tests := []struct {
		name string
		x    [2]float64
		want bool
	}{
		{name: "0 and 0 == false", x: [2]float64{0, 0}, want: false},
		{name: "1 and 0 == false", x: [2]float64{1, 0}, want: false},
		{name: "0 and 1 == false", x: [2]float64{0, 1}, want: false},
		{name: "1 and 1 == false", x: [2]float64{1, 1}, want: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			x := mat.NewVecDense(2, tc.x[:])
			got := cpu.And(x)

			if got != tc.want {
				t.Fatalf("expected: %v, got: %v", tc.want, got)
			}
		})
	}
}
