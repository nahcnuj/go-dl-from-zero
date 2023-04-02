//go:build cpu

package chap2

import (
	"fmt"
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
)

func TestAndByPerceptron(t *testing.T) {
	cpu := calculator.NewCPUBackend()

	tests := []struct {
		x, y float64
		want bool
	}{
		{x: 0, y: 0, want: false},
		{x: 1, y: 0, want: false},
		{x: 0, y: 1, want: false},
		{x: 1, y: 1, want: true},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%v ∧ %v == %v", tc.x, tc.y, tc.want), func(t *testing.T) {
			got, err := And(cpu, tc.x, tc.y)
			if err != nil {
				t.Fatal(err)
			}

			if got != tc.want {
				t.Errorf("expected: %v, got: %v", tc.want, got)
			}
		})
	}
}