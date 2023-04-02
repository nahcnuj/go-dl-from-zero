package main

import (
	"github.com/nahcnuj/go-dl-from-zero/calculator"
	"github.com/nahcnuj/go-dl-from-zero/chap2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg/draw"
)

func plotXor() *plot.Plot {
	cpu := calculator.NewCPUBackend()

	truePoints := make(plotter.XYs, 0)
	falsePoints := make(plotter.XYs, 0)

	for x1 := -5; x1 < 15; x1++ {
		for x2 := -5; x2 < 15; x2++ {
			x1, x2 := float64(x1)/float64(10), float64(x2)/float64(10)
			y, _ := chap2.Xor(cpu, x1, x2)

			p := plotter.XY{X: x1, Y: x2}
			if y > 0 {
				truePoints = append(truePoints, p)
			} else {
				falsePoints = append(falsePoints, p)
			}
		}
	}

	p := plot.New()
	p.Title.Text = "NAND by perceptrons"
	p.X.Label.Text = "x1"
	p.Y.Label.Text = "x2"

	p.Add(plotter.NewGrid())

	ts, _ := plotter.NewScatter(truePoints)
	ts.GlyphStyle.Color = plotutil.Color(0)
	ts.Shape = draw.CircleGlyph{}
	p.Add(ts)
	p.Legend.Add("true", ts)

	fs, _ := plotter.NewScatter(falsePoints)
	fs.GlyphStyle.Color = plotutil.Color(1)
	fs.Shape = draw.CircleGlyph{}

	p.Add(fs)
	p.Legend.Add("false", fs)

	return p
}
