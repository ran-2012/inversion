#pragma once

#include <sstream>

#include <cuda_runtime.h>

#include ""
using float_t = global::float_t;
using floath_ptr = std::unique_ptr<float_t[]>;

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