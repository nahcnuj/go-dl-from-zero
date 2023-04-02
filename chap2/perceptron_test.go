package chap2

import (
	"fmt"
	"testing"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
)

func testAnd[T calculator.Scalar](t *testing.T, be calculator.Backend[T]) {
	tests := []struct {
		x, y T
		want bool
	}{
		{x: 0, y: 0, want: false},
		{x: 1, y: 0, want: false},
		{x: 0, y: 1, want: false},
		{x: 1, y: 1, want: true},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%v ∧ %v == %v", tc.x, tc.y, tc.want), func(t *testing.T) {
			got, err := And(be, tc.x, tc.y)
			if err != nil {
				t.Fatal(err)
			}

			if got != tc.want {
				t.Errorf("expected: %v, got: %v", tc.want, got)
			}
		})
	}
}

func testNand[T calculator.Scalar](t *testing.T, be calculator.Backend[T]) {
	tests := []struct {
		x, y T
		want bool
	}{
		{x: 0, y: 0, want: true},
		{x: 1, y: 0, want: true},
		{x: 0, y: 1, want: true},
		{x: 1, y: 1, want: false},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%v ∧ %v == %v", tc.x, tc.y, tc.want), func(t *testing.T) {
			got, err := Nand(be, tc.x, tc.y)
			if err != nil {
				t.Fatal(err)
			}

			if got != tc.want {
				t.Errorf("expected: %v, got: %v", tc.want, got)
			}
		})
	}
}

func testOr[T calculator.Scalar](t *testing.T, be calculator.Backend[T]) {
	tests := []struct {
		x, y T
		want bool
	}{
		{x: 0, y: 0, want: false},
		{x: 1, y: 0, want: true},
		{x: 0, y: 1, want: true},
		{x: 1, y: 1, want: true},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%v ∧ %v == %v", tc.x, tc.y, tc.want), func(t *testing.T) {
			got, err := Or(be, tc.x, tc.y)
			if err != nil {
				t.Fatal(err)
			}

			if got != tc.want {
				t.Errorf("expected: %v, got: %v", tc.want, got)
			}
		})
	}
}
