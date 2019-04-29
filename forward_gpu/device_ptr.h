#pragma once

#include <cmath>
#include <cassert>
#include <sstream>
#include <memory>

#include <cuda_runtime.h>

#include "cuda_helper.h"

namespace gpu
{
	/**
	 * \brief device数据智能指针，自动释放显存
	 * \tparam T 数据类型
	 */
	template <typename T = float_t>
	class device_ptr
	{
	private:
		T* device_mem;

	public:
		device_ptr() { device_mem = nullptr; }
		device_ptr(const device_ptr<T>& p) = delete;

		device_ptr(device_ptr<T>&& p) noexcept : device_mem(p.device_mem)
		{
			p.device_mem = nullptr;
		}

		~device_ptr()
		{
			release();
		}

		device_ptr<T>& operator=(device_ptr<T>&& p) noexcept
		{
			this->release();
			device_mem = p.get();
			p.device_mem = nullptr;

			return *this;
		}

		/**
		 * \brief 分配显存
		 * \param size 显存大小
		 */
		void allocate(size_t size)
		{
			release();
			auto err = cudaMalloc(&device_mem, size * sizeof(T));
			CHECK;
		}

		__host__ __device__ T* get() const
		{
			return device_mem;
		}

		/**
		 * \brief 释放显存
		 */
		void release() noexcept
		{
			if (!device_mem)
			{
				return;
			}
			auto err = cudaFree(device_mem);
			device_mem = nullptr;
		}
	};
}
