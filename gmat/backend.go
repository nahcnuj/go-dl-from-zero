package gmat

import (
	"errors"
	"fmt"
	"os"

	"github.com/PassKeyRa/go-opencl/opencl"
)

const programCode = `
kernel void vec_add(global float *out, global float *a, global float *b)
{
	size_t i = get_global_id(0);
	out[i] = a[i] + b[i];
}

kernel void vec_dot(global float *out, global float *a, global float *b)
{
	size_t gid = get_global_id(0);
	out[gid] = a[gid] * b[gid];
}
`

func getKernelNames() []string {
	return []string{
		"vec_add",
		"vec_dot",
	}
}

type Backend struct {
	device  opencl.Device
	context opencl.Context
	kernels map[string]opencl.Kernel
}

func NewBackend() (backend *Backend, err error) {
	device, err := getDevice()
	if err != nil {
		return
	}

	context, err := device.CreateContext()
	if err != nil {
		return
	}

	backend = &Backend{
		device:  *device,
		context: context,
		kernels: make(map[string]opencl.Kernel),
	}

	if err = backend.buildKernels(); err != nil {
		backend.Release()
		return
	}

	return
}

func (b *Backend) Release() {
	for _, k := range b.kernels {
		k.Release()
	}
	b.context.Release()
	b = nil
}

func (b *Backend) buildKernels() error {
	program, err := b.context.CreateProgramWithSource(programCode)
	if err != nil {
		return err
	}
	defer program.Release()

	var log string
	if err := program.Build(b.device, &log); err != nil || len(log) > 0 {
		fmt.Fprintln(os.Stderr, log)
		return err
	}

	for _, name := range getKernelNames() {
		kernel, err := program.CreateKernel(name)
		if err != nil {
			return err
		}
		b.kernels[name] = kernel
	}
	return nil
}

func getDevice() (*opencl.Device, error) {
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
