//go:build gpu

package main

import (
	"flag"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"

	"github.com/nahcnuj/go-dl-from-zero/gmat"
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

	gpu, err := gmat.NewBackend()
	if err != nil {
		panic(err)
	}
	defer gpu.Release()

	a, b := make([]float32, dataSize), make([]float32, dataSize)
	for cnt := 0; cnt < 1_000; cnt++ {
		for i := 0; i < dataSize; i++ {
			a[i] = rand.Float32()
			b[i] = rand.Float32()
		}

		va, vb := gpu.NewVecDense(a), gpu.NewVecDense(b)

		gpu.Dot(va, vb)
	}

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
