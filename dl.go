package main

import "go-dl-from-zero/mat"

type DeepLearning struct {
	mat.Backend
}

func NewDeepLearning(backend mat.Backend) DeepLearning {
	return DeepLearning{Backend: backend}
}

func (dl *DeepLearning) Release() {
	dl.Backend.Release()
}
