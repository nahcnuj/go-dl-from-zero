//go:build gpu

package calculator_test

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	"github.com/nahcnuj/go-dl-from-zero/calculator"

	"github.com/stretchr/testify/assert"
)

func TestAddVectors(t *testing.T) {
	tests := []struct {
		x    []float32
		y    []float32
		want []float32
	}{
		{x: []float32{1, 2, 3}, y: []float32{4, 5, 6}, want: []float32{5, 7, 9}},
		{x: []float32{1, 2, 3}, y: []float32{-1, -2, -3}, want: []float32{0, 0, 0}},
		{x: []float32{1, -2, 3}, y: []float32{-1, 2, -3}, want: []float32{0, 0, 0}},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%v + %v = %v", tc.x, tc.y, tc.want), func(t *testing.T) {
			x, y := gpu.NewVector(tc.x), gpu.NewVector(tc.y)
			want := gpu.NewVector(tc.want)

			got, err := gpu.AddVectors(x, y)
			if err != nil {
				t.Fatal(err)
			}

			assert.InDeltaSlice(t, want.Raw(), got.Raw(), delta)
		})
	}

	assertion := func(v *twoSameHugeDimGPUVectors) bool {
		got, err := gpu.AddVectors(v.A, v.B)
		if err != nil {
			t.Log(err)
			return false
		}

		return assert.InDeltaSlice(t, v.WantAdd.Raw(), got.Raw(), delta)
	}
	if err := quick.Check(assertion, nil); err != nil {
		t.Fatal(err)
	}

}

// func TestAddVectorsProperties(t *testing.T) {
// 	t.Run("a + (-a) == 0", func(t *testing.T) {
// 		assertion := func(a *TestGPUVector) bool {
// 			dim := a.Dim()

// 			tb := make([]float32, dim)
// 			for i := 0; i < dim; i++ {
// 				tb[i] = -a.AtVec(i)
// 			}
// 			b := gpu.NewVecDense(tb)

// 			got, err := gpu.AddVectors(a.VecDense, b)
// 			if err != nil {
// 				panic(err)
// 			}

// 			for i := 0; i < got.Len(); i++ {
// 				got := got.AtVec(i)
// 				if ok := assert.InDelta(t, 0, got, delta); !ok {
// 					return false
// 				}
// 			}
// 			return true
// 		}

// 		if err := quick.Check(assertion, nil); err != nil {
// 			t.Fatal(err)
// 		}
// 	})
// }

func TestDot(t *testing.T) {
	tests := []struct {
		x    []float32
		y    []float32
		want float32
	}{
		{x: []float32{1, 2, 3}, y: []float32{4, 5, 6}, want: 4 + 10 + 18},
		{x: []float32{1, 2, 3}, y: []float32{-1, -2, -3}, want: -1 - 4 - 9},
		{x: []float32{1, -2, 3}, y: []float32{-1, 2, -3}, want: -1 - 4 - 9},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%v + %v = %v", tc.x, tc.y, tc.want), func(t *testing.T) {
			x, y := gpu.NewVector(tc.x), gpu.NewVector(tc.y)

			got, err := gpu.Dot(x, y)
			if err != nil {
				t.Fatal(err)
			}

			if got != tc.want {
				t.Errorf("got %v, expected %v", got, tc.want)
			}
		})
	}

	assertion := func(v *twoSameHugeDimGPUVectors) bool {
		got, err := gpu.Dot(v.A, v.B)
		if err != nil {
			t.Log(err)
			return false
		}
		return assert.InEpsilon(t, v.WantDot, got, epsilon)
	}
	if err := quick.Check(assertion, nil); err != nil {
		t.Fatal(err)
	}
}

func TestVectorElementWiseGreaterThan(t *testing.T) {
	tests := []struct {
		x, y []float32
		want []float32
	}{
		{x: []float32{1, 1, 1}, y: []float32{0, 0, 0}, want: []float32{1, 1, 1}},
		{x: []float32{0, 0, 0}, y: []float32{0, 0, 0}, want: []float32{0, 0, 0}},
		{x: []float32{1e-6, 0, -1e-6}, y: []float32{0, 0, 0}, want: []float32{1, 0, 0}},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%v > %v = %v", tc.x, tc.y, tc.want), func(t *testing.T) {
			x, y := gpu.NewVector(tc.x), gpu.NewVector(tc.y)

			got, err := gpu.VectorElementWiseGreaterThan(x, y)
			if err != nil {
				t.Fatal(err)
			}

			assert.InDeltaSlice(t, tc.want, got.Raw(), delta)
		})
	}

	assertion := func(v *twoSameHugeDimGPUVectors) bool {
		got, err := gpu.VectorElementWiseGreaterThan(v.A, v.B)
		if err != nil {
			t.Log(err)
			return false
		}
		return assert.InDeltaSlice(t, v.WantGreaterThan.Raw(), got.Raw(), delta)
	}
	if err := quick.Check(assertion, nil); err != nil {
		t.Fatal(err)
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		x    []float32
		want []float32
	}{
		{x: []float32{0}, want: []float32{0.5}},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("sigmoid(%v) = %v", tc.x, tc.want), func(t *testing.T) {
			x := gpu.NewVector(tc.x)

			got, err := gpu.Sigmoid(x)
			if err != nil {
				t.Fatal(err)
			}

			assert.InDeltaSlice(t, tc.want, got.Raw(), delta)
		})
	}

	assertion := func(v *twoSameHugeDimGPUVectors) bool {
		got, err := gpu.Sigmoid(v.A)
		if err != nil {
			t.Log(err)
			return false
		}
		return assert.InDeltaSlice(t, v.WantSigmoid.Raw(), got.Raw(), delta)
	}
	if err := quick.Check(assertion, nil); err != nil {
		t.Fatal(err)
	}
}

func TestReLU(t *testing.T) {
	tests := []struct {
		x    []float32
		want []float32
	}{
		{x: []float32{-1e6, -1, -1e-6, 0, 1e-6, 1, 1e6}, want: []float32{0, 0, 0, 0, 1e-6, 1, 1e6}},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("ReLU(%v) = %v", tc.x, tc.want), func(t *testing.T) {
			x := gpu.NewVector(tc.x)

			got, err := gpu.ReLU(x)
			if err != nil {
				t.Fatal(err)
			}

			assert.InDeltaSlice(t, tc.want, got.Raw(), delta)
		})
	}

	assertion := func(v *twoSameHugeDimGPUVectors) bool {
		got, err := gpu.ReLU(v.A)
		if err != nil {
			t.Log(err)
			return false
		}
		return assert.InDeltaSlice(t, v.WantReLU.Raw(), got.Raw(), delta)
	}
	if err := quick.Check(assertion, nil); err != nil {
		t.Fatal(err)
	}
}

type TestGPUVector struct {
	*calculator.GPUVector
}

func (*TestGPUVector) Generate(rand *rand.Rand, size int) reflect.Value {
	n := rand.Intn(size-1) + 1
	t := make([]float32, n)
	for i := 0; i < n; i++ {
		t[i] = rand.Float32()
	}
	return reflect.ValueOf(&TestGPUVector{gpu.NewVector(t).(*calculator.GPUVector)})
}

func BenchmarkDot(b *testing.B) {
	assertion := func(v *twoSameHugeDimGPUVectors) bool {
		got, err := gpu.Dot(v.A, v.B)
		if err != nil {
			b.Log(err)
			return false
		}
		return assert.InEpsilon(b, v.WantDot, got, epsilon)
	}

	b.ResetTimer()
	if err := quick.Check(assertion, &quick.Config{
		MaxCount: b.N,
	}); err != nil {
		b.Fatal(err)
	}
}

const hugeDim = 1024

type twoSameHugeDimGPUVectors struct {
	A, B            calculator.Vector[float32]
	WantAdd         calculator.Vector[float32]
	WantDot         float32
	WantGreaterThan calculator.Vector[float32]
	WantSigmoid     calculator.Vector[float32]
	WantReLU        calculator.Vector[float32]
}

func (*twoSameHugeDimGPUVectors) Generate(rand *rand.Rand, size int) reflect.Value {
	var (
		n       = hugeDim
		a, b    = make([]float32, n), make([]float32, n)
		add     = make([]float32, n)
		dot     float32
		gt      = make([]float32, n)
		sigmoid = make([]float32, n)
		relu    = make([]float32, n)
	)
	for i := 0; i < n; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
		add[i] = a[i] + b[i]
		dot += a[i] * b[i]
		if a[i] > b[i] {
			gt[i] = 1
		}
		sigmoid[i] = float32(1 / (1 + math.Exp(float64(-a[i]))))
		relu[i] = float32(math.Max(0, float64(a[i])))
	}
	return reflect.ValueOf(&twoSameHugeDimGPUVectors{
		A:               gpu.NewVector(a),
		B:               gpu.NewVector(b),
		WantAdd:         gpu.NewVector(add),
		WantDot:         dot,
		WantGreaterThan: gpu.NewVector(gt),
		WantSigmoid:     gpu.NewVector(sigmoid),
		WantReLU:        gpu.NewVector(relu),
	})
}
