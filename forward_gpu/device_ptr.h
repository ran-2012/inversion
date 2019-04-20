#pragma once

#include <cmath>
#include <cassert>
#include <sstream>
#include <memory>
#include <exception>

#include <cuda_runtime.h>

#include "../global/global.h"

//device数据智能指针，自动释放显存
class device_ptr
{
public:
	using floatd_t = global::float_t;

private:
	floatd_t *device_mem;

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
		assert(!device_mem);
		cudaMalloc(&device_mem, size * sizeof(floatd_t));
	}
	floatd_t* get()
	{
		return device_mem;
	}
	void release() noexcept
	{
		assert(device_mem);
		cudaFree(device_mem);
		device_mem = nullptr;
	}
};