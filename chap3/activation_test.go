package chap3

import (
	"fmt"
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
	"github.com/stretchr/testify/assert"
)

const (
	delta = 1e-5
)

func testStep[T calculator.Scalar](t *testing.T, be calculator.Backend[T]) {
	tests := []struct {
		x    calculator.Vector[T]
		want calculator.Vector[T]
	}{
		{x: be.NewVector([]T{-1, 0, 1}), want: be.NewVector([]T{0, 0, 1})},
		{x: be.NewVector([]T{-1e-6, 0, 1e-6}), want: be.NewVector([]T{0, 0, 1})},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("step(%v) = %v", tc.x.Raw(), tc.want.Raw()), func(t *testing.T) {
			got, err := be.VectorElementWiseGreaterThan(tc.x, be.ZeroVector(tc.x.Dim()))
			if err != nil {
				t.Fatal(err)
			}

			assert.InDeltaSlice(t, tc.want.Raw(), got.Raw(), delta)
		})
	}
}
