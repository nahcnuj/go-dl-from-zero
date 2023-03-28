//go:build gpu

package mat

import (
	"errors"
	"fmt"
	"os"

	"github.com/PassKeyRa/go-opencl/opencl"
)

type GPUBackend struct {
	device *GPUDevice
}

func NewGPUBackend() (Backend, error) {
	platforms, err := opencl.GetPlatforms()
	if err != nil {
		return nil, err
	}

	var device *GPUDevice
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
				device, err = NewGPUDevice(d)
				if err != nil {
					return nil, err
				}
				return &GPUBackend{device: device}, nil
			}
		}
	}
	return nil, errors.New("no GPU devices available")
}

func (gpu *GPUBackend) Release() {
	gpu.device.Release()
}

const programCode = `
kernel void vec_add(global float *out, global float *a, global float *b)
{
	size_t i = get_global_id(0);
	out[i] = a[i] + b[i];
}
`

func getKernelNames() []string {
	return []string{
		"vec_add",
	}
}

type GPUDevice struct {
	device  opencl.Device
	context opencl.Context
	kernels map[string]opencl.Kernel
	queue   opencl.CommandQueue
}

func NewGPUDevice(d opencl.Device) (device *GPUDevice, err error) {
	c, err := d.CreateContext()
	if err != nil {
		return nil, err
	}

	q, err := c.CreateCommandQueue(d)
	if err != nil {
		c.Release()
		return nil, err
	}

	device = &GPUDevice{
		device:  d,
		context: c,
		kernels: make(map[string]opencl.Kernel),
		queue:   q,
	}
	if err = device.createKernels(); err != nil {
		device.Release()
		return nil, err
	}
	return
}

func (d *GPUDevice) Release() {
	d.context.Release()
	d.queue.Release()
	for _, k := range d.kernels {
		k.Release()
	}
}

func (d *GPUDevice) createKernels() error {
	program, err := d.context.CreateProgramWithSource(programCode)
	if err != nil {
		return err
	}
	defer program.Release()

	var log string
	if err := program.Build(d.device, &log); err != nil || len(log) > 0 {
		fmt.Fprintln(os.Stderr, log)
		return err
	}

	for _, name := range getKernelNames() {
		kernel, err := program.CreateKernel(name)
		if err != nil {
			return err
		}
		d.kernels[name] = kernel
	}
	return nil
}
