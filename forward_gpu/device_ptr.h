#pragma once

#include <cmath>
#include <cassert>
#include <sstream>
#include <memory>
#include <exception>

#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "../global/global.h"

//device数据智能指针，自动释放显存
class device_ptr
{
private:
	float_t *device_mem;

public:
	device_ptr() { device_mem = nullptr; }
	device_ptr(const device_ptr& p) = delete;
	device_ptr(device_ptr&& p) noexcept :device_mem(p.device_mem)
	{
		p.device_mem = nullptr;
	}

	~device_ptr()
	{
		release();
	}

	device_ptr& operator=(device_ptr&& p) noexcept
	{
		this->release();
		device_mem = p.get();
		p.release();
		return *this;
	}

	void allocate(size_t size)
	{
		auto err = cudaMalloc(&device_mem, size * sizeof(float_t));
		CHECK;
	}
	float_t* get()
	{
		return device_mem;
	}
	void release() noexcept
	{
		if (!device_mem)
		{
			return;
		}
		auto err = cudaFree(device_mem);
		CHECK;
		device_mem = nullptr;
	}
};