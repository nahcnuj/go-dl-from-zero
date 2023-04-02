//go:build cpu

package calculator_test

import (
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	"github.com/nahcnuj/go-dl-from-zero/calculator"
	"github.com/stretchr/testify/assert"
)

func BenchmarkDotVecByCPU(b *testing.B) {
	assertion := func(v *twoSameHugeDimCPUVector) bool {
		got, _ := cpu.Dot(v.A, v.B)
		return assert.InEpsilon(b, v.WantDot, got, epsilon)
	}

	b.ResetTimer()
	if err := quick.Check(assertion, &quick.Config{
		MaxCount: b.N,
	}); err != nil {
		b.Fatal(err)
	}
}

type twoSameHugeDimCPUVector struct {
	A, B    *calculator.CPUVector
	WantDot float64
}

const hugeDim = 1024

func (*twoSameHugeDimCPUVector) Generate(rand *rand.Rand, size int) reflect.Value {
	n := hugeDim
	a, b, ret := make([]float64, n), make([]float64, n), 0.0
	for i := 0; i < n; i++ {
		a[i] = rand.Float64()
		b[i] = rand.Float64()
		ret += a[i] * b[i]
	}
	return reflect.ValueOf(&twoSameHugeDimCPUVector{
		A:       cpu.NewVector(a).(*calculator.CPUVector),
		B:       cpu.NewVector(b).(*calculator.CPUVector),
		WantDot: ret,
	})
}
