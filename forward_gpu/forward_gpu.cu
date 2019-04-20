
#include <cmath>
#include <sstream>
#include <memory>
#include <exception>

#include <cuda_runtime.h>

#include "forward_gpu.h"
#include "device_ptr.h"
#include "../data/data.h"
#include "../global/global.h"

//cuda错误代码检查
#define CHECK_CUDA_ERROR(err)\
	if(err!=cudaSuccess)\
	{\
		std::stringstream msg;\
		msg << "cuda error: \n";\
		msg << cudaGetErrorName(err) << '\n';\
		msg << "at line: " << __LINE__;\
		throw std::runtime_error(msg.str());\
	}
//对err进行错误检查，需定义err为cudaError_t
#define CHECK CHECK_CUDA_ERROR(err)
//复制host内存内容到device中
void copy_to_device(void* host, void* device, size_t size)
{
	auto err = cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
	CHECK
}
//复制device显存内容到host中
void copy_to_host(void* device, const void* host, size_t size)
{
	auto err = cudaMemcpy(device, host, size, cudaMemcpyDeviceToHost);
	CHECK
}

//host中的数据
using floath_t=global::float_t;
using floath_ptr=std::unique_ptr<floath_t[]>;
//device中的数据
using floatd_t=global::float_t;

void forward_gpu::init_cuda_device()
{
	int device_count;
	auto err = cudaGetDeviceCount(&device_count);
	CHECK;
	global::log("forward", "init_cuda_device complete");
}

__global__ void test_device_kernel(floatd_t *a, floatd_t *b, floatd_t *c, int num)
{
	int i = threadIdx.x;
	if (i >= num)
	{
		return;
	}
	c[i] = a[i] * b[i];
}

void forward_gpu::test_cuda_device()
{
	device_ptr da;
	device_ptr db;
	device_ptr dc;

	floath_t a[] = { 1,2,3,4,5 };
	floath_t b[] = { 1,2,3,4,5 };

	auto size = sizeof(a) / sizeof(floath_t);
	floath_ptr c(new floath_t[size]);

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