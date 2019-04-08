
#include <sstream>
#include <exception>

#include <cuda_runtime.h>

#include "forward_gpu.h"
#include "../data/data.h"
#include "../data/global.h"

//host中的数据
using floath_t=global::float_t;
//device中的数据
using floatd_t=global::float_t;

using forward_data = forward_gpu::forward_data;

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
//复制host内存内容到device中，并检查错误
#define COPY_TO_DEVICE(host, device, size) err = cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);\
	CHECK
//复制device显存内容到host中，并检查错误
#define COPY_TO_HOST(device, host, size) err= cudaMemcpy(device, host, size, cudaMemcpyDeviceToHost);\
	CHECK

void forward_gpu::init_cuda_device()
{
	int device_count;
	auto err = cudaGetDeviceCount(&device_count);
	CHECK_CUDA_ERROR(err);
	
}

__global__ void test_device(floatd_t *a, floatd_t *b, floatd_t *c, int num)
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
	floatd_t *da;
	floatd_t *db;
	floatd_t *dc;

	floath_t a[] = { 1,2,3,4,5 };
	floath_t b[] = { 1,2,3,4,5 };
	auto size = sizeof(a);
	auto num = size / sizeof(floath_t);
	floath_t *c = new floath_t[size/sizeof(floath_t)];

	auto err=cudaMalloc(&da, size);
	CHECK
	err = cudaMalloc(&db, size);
	CHECK
	err = cudaMalloc(&dc, size);
	CHECK;

	COPY_TO_DEVICE(a, da, size);
	COPY_TO_DEVICE(b, db, size);

	test_device <<<32, 1 >>> (da, db, dc, num);

	COPY_TO_HOST(dc, c, size);
	for (auto i = 0; i < num; ++i)
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