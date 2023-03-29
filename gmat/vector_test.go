//go:build gpu

package gmat_test

import (
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"testing"
	"testing/quick"

	"go-dl-from-zero/gmat"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

const (
	delta   = 1e-5 // 許容絶対誤差
	epsilon = 0.1  // 許容相対誤差
)

var gpu *gmat.Backend

func TestMain(m *testing.M) {
	var err error
	gpu, err = gmat.NewBackend()
	if err != nil {
		panic(err)
	}
	defer gpu.Release()

	ret := m.Run()
	os.Exit(ret)
}

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
			x, y := gpu.NewVecDense(tc.x), gpu.NewVecDense(tc.y)
			want := gpu.NewVecDense(tc.want)

			got, err := gpu.AddVectors(x, y)
			if err != nil {
				t.Fatal(err)
			}

			assertEqualVectors(t, got, want)
		})
	}

	assertion := func(v *twoSameHugeDimGmatVector) bool {
		got, err := gpu.AddVectors(v.A, v.B)
		if err != nil {
			t.Log(err)
			return false
		}
		assertEqualVectors(t, got, v.WantAdd)
		return true
	}
	if err := quick.Check(assertion, nil); err != nil {
		t.Fatal(err)
	}

}

func TestAddVectorsProperties(t *testing.T) {
	t.Run("a + (-a) == 0", func(t *testing.T) {
		assertion := func(a *TestVecDense) bool {
			dim := a.Len()

			tb := make([]float32, dim)
			for i := 0; i < dim; i++ {
				tb[i] = -a.AtVec(i)
			}
			b := gpu.NewVecDense(tb)

			got, err := gpu.AddVectors(a.VecDense, b)
			if err != nil {
				panic(err)
			}

			for i := 0; i < got.Len(); i++ {
				got := got.AtVec(i)
				if ok := assert.InDelta(t, 0, got, delta, "want %.8f, got %.8f", 0, got); !ok {
					return false
				}
			}
			return true
		}

		if err := quick.Check(assertion, nil); err != nil {
			t.Fatal(err)
		}
	})
}

func TestDotVec(t *testing.T) {
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
			x, y := gpu.NewVecDense(tc.x), gpu.NewVecDense(tc.y)

			got, err := gpu.Dot(x, y)
			if err != nil {
				t.Fatal(err)
			}

			if got != tc.want {
				t.Errorf("got %v, expected %v", got, tc.want)
			}
		})
	}

	var n, c int
	assertion := func(v *twoSameHugeDimGmatVector) bool {
		got, err := gpu.Dot(v.A, v.B)
		if err != nil {
			t.Log(err)
			return false
		}
		ok := assert.InEpsilon(t, v.WantDot, got, epsilon, "want %.8f, got %.8f, relerror %20.8f", v.WantDot, got, (v.WantDot-got)/v.WantDot)
		if !ok {
			c++
		}
		n++
		return true
	}
	if err := quick.Check(assertion, nil); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d NG of %d tries", c, n)
}

type TestVecDense struct {
	*gmat.VecDense
}

func (*TestVecDense) Generate(rand *rand.Rand, size int) reflect.Value {
	n := randVectorSize(size)
	t := make([]float32, n)
	for i := 0; i < n; i++ {
		t[i] = rand.Float32()
	}
	v := &TestVecDense{gpu.NewVecDense(t)}
	return reflect.ValueOf(v)
}

func randVectorSize(max int) int {
	return rand.Intn(max-1) + 1
}

func assertEqualVectors(t testing.TB, got, want gmat.Vector) {
	t.Helper()

	if got.Len() != want.Len() {
		t.Errorf("dimension got: %v, expected: %v", got.Len(), want.Len())
	}
	for i := 0; i < got.Len(); i++ {
		assert.InDelta(t, want.AtVec(i), got.AtVec(i), delta, "[%d] want %v, got %v", i, want.AtVec(i), got.AtVec(i))
	}
}

const hugeDim = 1024

type twoSameHugeDimGmatVector struct {
	A       gmat.Vector
	B       gmat.Vector
	WantAdd gmat.Vector
	WantDot float32
}

func (*twoSameHugeDimGmatVector) Generate(rand *rand.Rand, size int) reflect.Value {
	n := hugeDim
	a, b, add, dot := make([]float32, n), make([]float32, n), make([]float32, n), float32(0)
	for i := 0; i < n; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
		add[i] = a[i] + b[i]
		dot += a[i] * b[i]
	}
	return reflect.ValueOf(&twoSameHugeDimGmatVector{
		A:       gpu.NewVecDense(a),
		B:       gpu.NewVecDense(b),
		WantAdd: gpu.NewVecDense(add),
		WantDot: dot,
	})
}

func BenchmarkDotVecByGPU(b *testing.B) {
	assertion := func(v *twoSameHugeDimGmatVector) bool {
		got, err := gpu.Dot(v.A, v.B)
		if err != nil {
			b.Log(err)
			return false
		}
		return assert.InEpsilon(b, v.WantDot, got, epsilon, "want %.8f, got %.8f, relerror %20.8f", v.WantDot, got, (v.WantDot-got)/v.WantDot)
	}

	b.ResetTimer()
	if err := quick.Check(assertion, &quick.Config{
		MaxCount: b.N,
	}); err != nil {
		b.Fatal(err)
	}
}

type twoSameHugeDimMatVector struct {
	A    mat.Vector
	B    mat.Vector
	Want float64
}

func (*twoSameHugeDimMatVector) Generate(rand *rand.Rand, size int) reflect.Value {
	n := hugeDim
	a, b, ret := make([]float64, n), make([]float64, n), 0.0
	for i := 0; i < n; i++ {
		a[i] = rand.Float64()
		b[i] = rand.Float64()
		ret += a[i] * b[i]
	}
	return reflect.ValueOf(&twoSameHugeDimMatVector{
		A:    mat.NewVecDense(n, a),
		B:    mat.NewVecDense(n, b),
		Want: ret,
	})
}

func BenchmarkDotVecByCPU(b *testing.B) {
	assertion := func(v *twoSameHugeDimMatVector) bool {
		got := mat.Dot(v.A, v.B)
		return assert.InEpsilon(b, v.Want, got, epsilon, "want %.8f, got %.8f, relerror %20.8f", v.Want, got, (v.Want-got)/v.Want)
	}

	b.ResetTimer()
	if err := quick.Check(assertion, &quick.Config{
		MaxCount: b.N,
	}); err != nil {
		b.Fatal(err)
	}
}
