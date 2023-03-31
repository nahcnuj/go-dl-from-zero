package main

import (
	"math"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func main() {
	x := make([]float64, 60)
	y1 := make([]float64, 60)
	y2 := make([]float64, 60)
	for i := 0; i < 6*10; i++ {
		x[i] = float64(i) / 10
		y1[i] = math.Sin(x[i])
		y2[i] = math.Cos(x[i])
	}

	p := plot.New()
	p.Title.Text = "sin and cos"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
	if err := plotutil.AddLinePoints(p,
		"y = sin(x)", convertToPlotterXY(x, y1),
		"y = cos(x)", convertToPlotterXY(x, y2)); err != nil {
		panic(err)
	}
	if err := p.Save(15*vg.Centimeter, 15*vg.Centimeter, "plot.png"); err != nil {
		panic(err)
	}
}

func convertToPlotterXY(x []float64, y []float64) (xy plotter.XYs) {
	xy = make(plotter.XYs, len(x))
	for i, _ := range x {
		xy[i] = plotter.XY{X: x[i], Y: y[i]}
	}
	return
}
