//go:build gpu

package mat_test

import (
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"testing"
	"testing/quick"

	"go-dl-from-zero/mat"
)

var gpu *mat.GPUBackend

func TestMain(m *testing.M) {
	be, err := mat.NewGPUBackend()
	if err != nil {
		panic(err)
	}
	gpu = be.(*mat.GPUBackend)
	defer gpu.Release()

	ret := m.Run()
	os.Exit(ret)
}

func TestAddVecByGPU(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
		y    []float64
		want []float64
	}{
		{x: []float64{1, 2, 3}, y: []float64{4, 5, 6}, want: []float64{5, 7, 9}},
		{x: []float64{1, 2, 3}, y: []float64{-1, -2, -3}, want: []float64{0, 0, 0}},
		{x: []float64{1, -2, 3}, y: []float64{-1, 2, -3}, want: []float64{0, 0, 0}},
	}

	for _, tc := range tests {
		name := tc.name
		if len(name) == 0 {
			name = fmt.Sprintf("%v + %v = %v", tc.x, tc.y, tc.want)
		}

		x, y := gpu.NewVector(tc.x), gpu.NewVector(tc.y)
		want := gpu.NewVector(tc.want)

		got := gpu.AddVectors(x, y)

		if got.Len() != want.Len() {
			t.Errorf("%s, dimension expected: %v, got: %v", name, want.Len(), got.Len())
		}
		for i := 0; i < got.Len(); i++ {
			if got.AtVec(i) != want.AtVec(i) {
				t.Errorf("%s, [%d] expected: %v, got: %v", name, i, want.AtVec(i), got.AtVec(i))
			}
		}
	}
}

func TestAddVectorsProperties(t *testing.T) {
	t.Run("a + (-a) == 0", func(t *testing.T) {
		t.Helper()
		assertion := func(a *gpuVector) bool {
			dim := a.Len()

			tb := make([]float64, dim)
			for i := 0; i < dim; i++ {
				tb[i] = -a.AtVec(i)
			}
			b := gpu.NewVector(tb)

			got := gpu.AddVectors(a.Vector, b)

			if got.Len() != dim {
				t.Logf("dimension expected: %v, got: %v", dim, got.Len())
				return false
			}
			for i := 0; i < dim; i++ {
				if got.AtVec(i) != 0 {
					t.Logf("[%d] expected: 0, got: %v", i, got.AtVec(i))
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

type gpuVector struct {
	mat.Vector
}

func (*gpuVector) Generate(rand *rand.Rand, size int) reflect.Value {
	n := randVectorSize(size)
	t := make([]float64, n)
	for i := 0; i < n; i++ {
		t[i] = randFloat64FromFloat32()
	}
	v := &gpuVector{gpu.NewVector(t)}
	return reflect.ValueOf(v)
}

func randVectorSize(max int) int {
	return rand.Intn(max-1) + 1
}

func randFloat64FromFloat32() float64 {
	return float64(rand.Float32())
}
