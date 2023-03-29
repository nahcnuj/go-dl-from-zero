//go:build gpu

package main

import (
	"flag"
	"go-dl-from-zero/mat"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
)

const dataSize = 1_000_000

var cpuProfile = flag.String("cpuprofile", "", "write CPU profile to `file`")
var memProfile = flag.String("memprofile", "", "write memory profile to `file`")

func main() {
	flag.Parse()
	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatal("could not create file for CPU profile: ", err)
		}
		defer f.Close()

		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("could not start CPU profiling: ", err)
		}
		defer pprof.StopCPUProfile()
	}

	gpu, err := mat.NewGPUBackend()
	if err != nil {
		panic(err)
	}
	defer gpu.Release()

	a, b := make([]float64, dataSize), make([]float64, dataSize)
	for cnt := 0; cnt < 1_000; cnt++ {
		for i := 0; i < dataSize; i++ {
			a[i] = float64(rand.Float32())
			b[i] = float64(rand.Float32())
		}

		va, vb := gpu.NewVector(a), gpu.NewVector(b)
		// fmt.Println("a  =", va)
		// fmt.Println("  b=", vb)

		gpu.Dot(va, vb)
	}
	// fmt.Println("a+b=", ret)

	if *memProfile != "" {
		f, err := os.Create(*memProfile)
		if err != nil {
			log.Fatal("could not create file for memory profile: ", err)
		}
		defer f.Close()

		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatal("could not write memory profile: ", err)
		}
	}
}
