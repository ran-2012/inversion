#pragma once

#include <cmath>
#include <cassert>
#include <sstream>
#include <memory>
#include <exception>

#include <cuda_runtime.h>

#include "../data/global.h"

//device数据智能指针，自动释放显存
class device_ptr
{
public:
	using float_t = global::float_t;

private:
	float_t *device_mem;

public:
	device_ptr() { device_mem = nullptr; }
	device_ptr(const device_ptr& p) = delete;
	device_ptr(device_ptr&& p)
	{
		*this = std::move(p);
	}

	~device_ptr()
	{
		release();
	}

	device_ptr operator=(device_ptr&& p)
	{
		this->release();
		device_mem = p.get();
		p.release();
	}

	void allocate(size_t size)
	{
		assert(!device_mem);
		cudaMalloc(&device_mem, size * sizeof(float_t));
	}
	float_t* get()
	{
		return device_mem;
	}
	void release() noexcept
	{
		if (device_mem)
		{
			cudaFree(device_mem);
			device_mem = nullptr;
		}
	}
};