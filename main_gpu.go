//go:build gpu

package main

import (
	"fmt"
	"go-dl-from-zero/mat"
	"math/rand"
)

const dataSize = 128

func main() {
	gpu, err := mat.NewGPUBackend()
	if err != nil {
		panic(err)
	}
	defer gpu.Release()

	a, b := make([]float64, dataSize), make([]float64, dataSize)
	for i := 0; i < dataSize; i++ {
		a[i] = float64(rand.Float32())
		b[i] = float64(rand.Float32())
	}

	va, vb := gpu.NewVector(a), gpu.NewVector(b)
	fmt.Println("a  =", va)
	fmt.Println("  b=", vb)

	ret := gpu.AddVectors(va, vb)
	fmt.Println("a+b=", ret)
}
