#pragma once

#include <cassert>
#include <vector>

#include "cuda_helper.h"
#include "device_ptr.h"

namespace gpu
{
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
		device_array() : device_mem_size(0)
		{
		}

		//复制vector中内存到显存中
		device_array(const vector& vec) : device_mem_size(vec.size())
		{
			device_mem.allocate(vec.size());
			copy_to_device(vec.data(), device_mem.get(), device_mem_size);

			allocate_device_this_ptr();
		}

		//分配指定大小显存
		device_array(size_t size) : device_mem_size(size)
		{
			device_mem.allocate(size);
			allocate_device_this_ptr();
		}

		__host__ device_array* get_device_ptr() const
		{
			return device_this_ptr.get();
		}

		__host__ void load_data(const vector& vec)
		{
			device_mem.release();
			device_mem.allocate(vec.size());
			copy_to_device(vec.data(), device_mem.get(), device_mem_size);
		}

		/**
		 * \brief 保存device中数据到vector中
		 * \param vec 目标vector
		 */
		__host__ void save_data(vector& vec) const
		{
			assert(device_mem.get());
			float_ptr host_mem(new float_t[device_mem_size]);

			copy_to_host(device_mem.get(), host_mem.get(), device_mem_size);

			vec.clear();
			vec.resize(size());
			for (auto i = 0; i < device_mem_size; ++i)
			{
				vec[i] = host_mem.get()[i];
			}
		}

		//访问数据指针，仅kernel中可访问
		__device__ float_t* get() const
		{
			return device_mem.get();
		}

		//访问数据，仅kernel中可访问
		__device__ float_t& operator[](size_t idx) const
		{
			assert(device_mem.get());
			return device_mem.get()[idx];
		}

		//获取大小，仅kernel中可访问
		__host__ __device__ size_t size() const
		{
			return device_mem_size;
		}
	};
}
