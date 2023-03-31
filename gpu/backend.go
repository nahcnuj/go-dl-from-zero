//go:build gpu

package gpu

import "github.com/nahcnuj/go-dl-from-zero/gmat"

type Backend struct {
	*gmat.Backend
}

func NewBackend() (*Backend, error) {
	be, err := gmat.NewBackend()
	if err != nil {
		return nil, err
	}
	return &Backend{be}, nil
}
