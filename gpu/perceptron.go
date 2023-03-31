//go:build gpu

package gpu

import "github.com/nahcnuj/go-dl-from-zero/gmat"

func (gpu *Backend) And(x gmat.Vector) (bool, error) {
	var (
		w         = gpu.NewVecDense([]float32{0.5, 0.5})
		b float32 = -0.7
	)
	dot, err := gpu.Dot(x, w)
	if err != nil {
		return false, err
	}
	return dot+b > 0, nil
}
