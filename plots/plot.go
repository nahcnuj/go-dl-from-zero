package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/vg"
)

var outDir = flag.String("out", ".", "output directory of plots")

func main() {
	flag.Parse()

	plots := map[string]func() *plot.Plot{
		"xor":        plotXor,
		"activation": plotActivationFuncs,
	}
	for name, f := range plots {
		path := filepath.Join(*outDir, fmt.Sprintf("%s.png", name))
		p := f()
		if err := p.Save(15*vg.Centimeter, 15*vg.Centimeter, path); err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
	}
}
