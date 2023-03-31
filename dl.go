package main

import "github.com/nahcnuj/go-dl-from-zero/gmat"

type DeepLearning struct {
	*gmat.Backend
}

func NewDeepLearning(backend *gmat.Backend) DeepLearning {
	return DeepLearning{Backend: backend}
}

func (dl *DeepLearning) Release() {
	dl.Backend.Release()
}
