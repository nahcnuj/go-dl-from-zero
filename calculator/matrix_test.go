package calculator_test

import (
	"fmt"
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
	"github.com/stretchr/testify/assert"
)

const (
	delta   = 1e-5 // 許容絶対誤差
	epsilon = 0.1  // 許容相対誤差
)

func testMultiply[T calculator.Scalar](t *testing.T, be calculator.Backend[T]) {
	tests := []struct {
		A, B [][]T
		Want [][]T
	}{
		{
			A:    [][]T{{1, 2}, {3, 4}},
			B:    [][]T{{5, 6}, {7, 8}},
			Want: [][]T{{19, 22}, {43, 50}},
		},
		{
			A:    [][]T{{1, 2, 3}, {4, 5, 6}},
			B:    [][]T{{1, 2}, {3, 4}, {5, 6}},
			Want: [][]T{{22, 28}, {49, 64}},
		},
		{
			A:    [][]T{{1, 2}, {3, 4}, {5, 6}},
			B:    [][]T{{7}, {8}},
			Want: [][]T{{23}, {53}, {83}},
		},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%v x %v = %v", tc.A, tc.B, tc.Want), func(t *testing.T) {
			got, err := be.Multiply(be.NewMatrix(tc.A), be.NewMatrix(tc.B))
			if err != nil {
				t.Fatal(err)
			}
			for r := 0; r < got.Rows(); r++ {
				assert.InDeltaSlice(t, tc.Want[r], got.RawRow(r), delta)
			}
		})
	}
}
