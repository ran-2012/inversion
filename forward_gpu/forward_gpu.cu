
#include <cmath>
#include <memory>
#include <exception>

#include <cuda_runtime.h>

#include "forward_gpu.h"
#include "device_ptr.h"
#include "cuda_helper.h"
#include "../data/data.h"
#include "../data/global.h"

void forward_gpu::init_cuda_device()
{
	int device_count;
	auto err = cudaGetDeviceCount(&device_count);
	CHECK;
	global::log("forward", "init_cuda_device complete");
}

__global__ void test_device_kernel(float_t *a, float_t *b, float_t *c, int num)
{
	int i = threadIdx.x;
	if (i >= num)
	{
		return;
	}
	c[i] = a[i] * b[i];
}

__global__ void hankel_transform(float_t a, float_t i0,
	float_t* hankel, float_t hankel_len,
	float_t* resistence, float_t res_len,
	floatd_*,)
{

}

void forward_gpu::test_cuda_device()
{
	device_ptr da;
	device_ptr db;
	device_ptr dc;

	float_t a[] = { 1,2,3,4,5 };
	float_t b[] = { 1,2,3,4,5 };

	auto size = sizeof(a) / sizeof(float_t);
	floath_ptr c(new float_t[size]);

	da.allocate(size);
	db.allocate(size);
	dc.allocate(size);

	copy_to_device(a, da.get(), size);
	copy_to_device(b, db.get(), size);

	test_device_kernel <<<32, 1 >>> (da, db, dc, size);

	copy_to_host(dc.get(), c.get(), size);

	for (auto i = 0; i < size; ++i)
	{
		if (a[i] * b[i] != c[i])
		{
			throw std::runtime_error("测试cuda设备失败，计算错误");
		}
	}
}

forward_data forward_gpu::forward()
{
	return forward_data();
}