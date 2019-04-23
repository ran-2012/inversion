#pragma once

#include <cmath>
#include <cassert>
#include <sstream>
#include <vector>
#include <exception>

#include <cuda_runtime.h>

#include "../global/global.h"
#include "cuda_helper.h"
#include "device_ptr.h"

//device数组
class device_array
{
private:
	device_ptr<float_t> device_mem;
	size_t device_mem_size;

	//在显存中的this
	device_ptr<device_array> device_this_ptr;

	__host__ void allocate_device_this_ptr()
	{
		device_this_ptr.release();
		device_this_ptr.allocate(1);
		copy_to_device(this, device_this_ptr.get(), 1);
	}

public:
	device_array() :device_mem_size(0)
	{

	}

	device_array(const std::vector<float_t>& vec) :device_mem_size(vec.size())
	{
		device_mem.allocate(vec.size());
		copy_to_device(vec.data(), device_mem.get(), device_mem_size);

		allocate_device_this_ptr();
	}

	__host__ device_array* get_device_ptr()
	{
		return device_this_ptr.get();
	}

	__host__ void load_data(const std::vector<float_t> &vec)
	{
		device_mem.release();
		device_mem.allocate(vec.size());
		copy_to_device(vec.data(), device_mem.get(), device_mem_size);
	}
	
	__host__ void save_data(std::vector<float_t>& vec)
	{
		assert(device_mem.get());
		float_ptr host_mem(new float_t(device_mem_size));

		copy_to_host(device_mem.get(), host_mem.get(), device_mem_size);

		vec.clear();
		vec.resize(size());
		for (auto i = 0; i < device_mem_size; ++i)
		{
			vec[i] = host_mem.get()[i];
		}
	}
	//访问数据指针，仅kernel中可访问
	__device__ float_t* get()
	{
		return device_mem.get();
	}
	//访问数据，仅kernel中可访问
	__device__ float_t& operator[](size_t idx)
	{
		assert(device_mem.get());
		return device_mem.get()[idx];
	}
	//获取大小，仅kernel中可访问
	__device__ size_t size()
	{
		return device_mem_size;
	}

};