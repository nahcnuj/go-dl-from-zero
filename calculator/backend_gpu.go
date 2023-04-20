//go:build gpu

package calculator

import (
	_ "embed"
	"errors"
	"fmt"
	"os"

	"github.com/bbedward/go-opencl/opencl"
)

//go:embed kernel.cl
var code string

func getKernelNames() []string {
	return []string{
		"vectorAdd",
		"vectorDot",
		"vectorElementWiseGreaterThan",
		"vectorSigmoid",
		"vectorReLU",
		"matrixMultiply",
	}
}

type GPUBackend struct {
	device  opencl.Device
	context opencl.Context
	kernels map[string]opencl.Kernel
}

var _ Backend[float32] = &GPUBackend{}

func NewGPUBackend() (Backend[float32], error) {
	device, err := getGPUDevice()
	if err != nil {
		return nil, err
	}

	context, err := device.CreateContext()
	if err != nil {
		return nil, err
	}

	backend := &GPUBackend{
		device:  *device,
		context: context,
		kernels: make(map[string]opencl.Kernel),
	}

	if err = backend.buildKernels(); err != nil {
		backend.Release()
		return nil, err
	}

	return backend, nil
}

func (gpu *GPUBackend) Release() {
	for _, k := range gpu.kernels {
		k.Release()
	}
	gpu.context.Release()
}

func (gpu *GPUBackend) buildKernels() error {
	program, err := gpu.context.CreateProgramWithSource(code)
	if err != nil {
		return err
	}
	defer program.Release()

	var log string
	if err := program.Build(gpu.device, &log); err != nil || len(log) > 0 {
		fmt.Fprintln(os.Stderr, log)
		return err
	}

	for _, name := range getKernelNames() {
		kernel, err := program.CreateKernel(name)
		if err != nil {
			return err
		}
		gpu.kernels[name] = kernel
	}
	return nil
}

func getGPUDevice() (*opencl.Device, error) {
	platforms, err := opencl.GetPlatforms()
	if err != nil {
		return nil, err
	}

	for _, platform := range platforms {
		var devices []opencl.Device
		devices, err = platform.GetDevices(opencl.DeviceTypeAll)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			continue
		}

		// Use the first available device
		for _, d := range devices {
			var available bool
			err = d.GetInfo(opencl.DeviceAvailable, &available)
			if err == nil && available {
				return &d, nil
			}
		}
	}

	return nil, errors.New("no GPU devices available")
}
