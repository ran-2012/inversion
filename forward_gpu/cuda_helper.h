#pragma once

#include <sstream>
#include <complex>

#include <cuda_runtime.h>

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

namespace gpu
{
	using float_t = global::float_t;
	using float_ptr=std::unique_ptr<float_t[]>;
	using vector=global::vector;

	/**
	 * \brief 复制host内存内容到device
	 * \tparam T 数据类型
	 * \param host host内存
	 * \param device device内存
	 * \param size 数据长度
	 */
	template <typename T>
	void copy_to_device(const T* host, T* device, size_t size)
	{
		auto err = cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice);
		CHECK
	}

	//
	/**
	 * \brief 复制device内容到host
	 * \tparam T 数据类型
	 * \param host host内存
	 * \param device device内存
	 * \param size 数据长度
	 */
	template <typename T>
	void copy_to_host(T* device, T* host, size_t size)
	{
		auto err = cudaMemcpy(host, device, size * sizeof(T), cudaMemcpyDeviceToHost);
		CHECK
	}
}
