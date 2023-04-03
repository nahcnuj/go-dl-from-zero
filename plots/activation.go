package main

import (
	"github.com/nahcnuj/go-dl-from-zero/calculator"
	"github.com/nahcnuj/go-dl-from-zero/chap3"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg/draw"
)

func plotActivationFuncs() *plot.Plot {
	cpu := calculator.NewCPUBackend()

	points := make(plotter.XYs, 0)

	for x := -50; x < 50; x++ {
		x := float64(x) / float64(10)
		y, _ := chap3.Step(cpu, cpu.NewVector([]float64{x}))

		p := plotter.XY{X: x, Y: y.Raw()[0]}
		points = append(points, p)
	}

	p := plot.New()
	p.Title.Text = "Activation Functions"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	p.Add(plotter.NewGrid())

	s, _ := plotter.NewScatter(points)
	s.GlyphStyle.Color = plotutil.Color(0)
	s.Shape = draw.CircleGlyph{}
	p.Add(s)
	p.Legend.Add("Step", s)

	return p
}
