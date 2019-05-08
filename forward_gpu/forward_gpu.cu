#include <cmath>
#include <memory>
#include <complex>

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include "forward_gpu.h"
#include "device_ptr.h"
#include "device_array.h"
#include "cuda_helper.h"

namespace gpu
{
	__device__ const float_t a1 = -7.91001919000e+00;
	__device__ const float_t s1 = 8.79671439570e-02;
	__device__ const float_t mu0 = global::mu0;
	__device__ const float_t pi = global::pi;

	__global__ void test_device_kernel(device_array* a, device_array* b, device_array* c)
	{
		int i = threadIdx.x;
		int n = a->size();

		auto da = a->get();
		auto db = b->get();
		auto dc = c->get();

		if (i >= n)
		{
			return;
		}
		printf("%f = %f * %f\n", dc[i], da[i], db[i]);
		dc[i] = da[i] * db[i];
	}

	__device__ thrust::complex<float_t> return_dHz_w(float_t a, float_t i0, float_t h,
	                                                 device_array* hankel,
	                                                 device_array* resistivity,
	                                                 device_array* height,
	                                                 thrust::complex<float_t> w)
	{
		using complex = thrust::complex<float_t>;

		complex ret(0, 0);

		const float_t* hankel_ptr = hankel->get();
		const float_t* res_ptr = resistivity->get();
		const float_t* height_ptr = height->get();

		const int res_size = resistivity->size();
		const int hankel_size = hankel->size();

		for (int k = 0; k < hankel_size; ++k)
		{
			const complex i(0, 1);
			const float_t lmd = 1 / a * powf(10, a1 + (k * s1));

			const complex u1 = sqrt(powf(lmd, 2) - i * w * mu0 / res_ptr[0]);

			complex r0 = 1;
			for (int cc = res_size - 2; cc > 00; --cc)
			{
				const float_t lmd_2 = pow(lmd, 2);
				const complex wi = i * w * mu0;

				const complex ui = sqrt(lmd_2 - wi / res_ptr[cc]);
				const complex uii = sqrt(lmd_2 - wi / res_ptr[cc + 1]);

				const complex ss = ui / uii * r0;
				const complex ex1 = exp(-2 * ui * height_ptr[cc]);
				const complex ctan1 = (1 + ex1) / (1 - ex1);

				r0 = (1 + ctan1 * ss) / (ctan1 + ss);
			}
			const complex f1 = 1 + (lmd - u1 / r0) / (lmd + u1 / r0) * exp(-2 * lmd * h);

			ret += f1 * lmd * hankel_ptr[k];
		}
		ret = ret * i0 / 2;
		return ret;
	}

	/**
	 * \brief 计算正演kernel函数
	 * \param a 回线半径(m)
	 * \param i0 发射电流(A)
	 * \param h 发射、接收回线高度(m)
	 * \param cosine 余弦变换系数
	 * \param hankel 汉克尔变换系数
	 * \param resistivity 地层电阻率
	 * \param height 地层厚度
	 * \param time 时间
	 * \param b 正演磁场强度
	 */
	__global__ void forward_kernel(float_t a, float_t i0, float_t h,
	                               device_array* cosine,
	                               device_array* hankel,
	                               device_array* resistivity,
	                               device_array* height,
	                               device_array* time,
	                               device_array* b)
	{
		const int time_idx = blockIdx.x;
		const int cosine_num = blockDim.x;
		const int cosine_idx = threadIdx.x;

		extern __shared__ float_t res[];
		__shared__ float_t t;
		__shared__ float_t* cosine_ptr;

		//每个block中的第一个线程为计算准备数据
		if (cosine_idx == 0)
		{
			t = time->get()[time_idx];
			cosine_ptr = cosine->get();
		}

		__syncthreads();

		float_t w = 1 / t * exp((-150 + cosine_idx - 1) * std::log(10.0) / 20);
		thrust::complex<float_t> hz_w = return_dHz_w(a, i0, h, hankel, resistivity, height, w);

		res[cosine_idx] = hz_w.imag() / w * cosine_ptr[cosine_idx];

		int sum_num = cosine_num;
		//二分求和
		while (sum_num > 0)
		{
			int idx = cosine_idx;
			int next_sum_num = sum_num / 2;

			//求和个数为奇数时会有一个多出来的元素
			if (sum_num | 0x1)
			{
				++idx;
				++next_sum_num;
			}
			const int offset = sum_num / 2;
			if (cosine_idx < offset)
			{
				res[idx] += res[idx + offset];
			}
			sum_num = next_sum_num;
			__syncthreads();
		}

		//保存正演结果磁场强度到全局显存
		if (cosine_idx == 0)
		{
			b->get()[time_idx] = res[0];
		}
	}

	/**
	 * \brief 计算响应kernel函数
	 * \param a 回线半径(m)
	 * \param i0 发射电流(A)
	 * \param h 发射、接收回线高度(m)
	 * \param b 磁场
	 * \param time 时间
	 * \param response_late_m 晚期磁场响应
	 * \param response_late_e 晚期电场响应
	 */
	__global__ void calc_response_kernel(float_t a, float_t i0, float_t h,
	                                     device_array* b,
	                                     device_array* time,
	                                     device_array* response_late_m,
	                                     device_array* response_late_e)
	{
		const int time_idx = threadIdx.x;

		const auto b_d = b->get();
		const auto t = time->get()[time_idx];

		const float_t dHz = sqrt(2 / pi) / t * b_d[time_idx];

		response_late_m->get()[time_idx] =
			mu0 * pow(pi * i0 * pow(a, 2) / 30 / abs(dHz), 2.0 / 3) / pi / t;

		const auto t1 = time->get()[time_idx + 1];
		const auto t2 = (t + t1) / 2;
		const auto bt = (b_d[time_idx + 1] - b_d[time_idx]) / (t2 - t1);

		response_late_e->get()[time_idx] =
			mu0 * pow(2 * pi * i0 * pow(a, 2) / 5 / t2 / abs(bt), 2.0 / 3) / 4 / pi / t2;
	}

	void init_cuda_device()
	{
		int device_count;
		auto err = cudaGetDeviceCount(&device_count);
		CHECK;
		global::log("forward", "init_cuda_device complete");
	}

	void test_cuda_device()
	{
		global::scoped_timer timer("test_cuda");

		global::log("test_cuda_device", "test start");

		device_array da;
		device_array db;
		device_array dc;

		vector a = {1, 2, 3, 4, 5};
		vector b = {1, 2, 3, 4, 5};
		vector c;

		const auto size = a.size();

		da.load_data(a);
		db.load_data(b);

		dc.allocate(size);

		global::log("test_cuda_device", "calculate start");

		test_device_kernel << <1, 32 >> >(da.get_device_ptr(), db.get_device_ptr(), dc.get_device_ptr());
		auto err = cudaDeviceSynchronize();
		CHECK;

		dc.save_data(c);

		global::log("test_cuda_device", "calculate end");

		for (auto i = 0; i < size; ++i)
		{
			if (a[i] * b[i] != c[i])
			{
				throw std::runtime_error("测试cuda设备失败，计算错误");
			}
		}
		global::log("test_cuda_device", "test end");
	}

	void forward(float_t a, float_t i0, float_t h,
	             const vector& cosine, const vector& hankel,
	             const vector& resistivity, const vector& height,
	             const vector& time,
	             vector& response_late_m, vector& response_late_e)
	{
		device_array cosine_d(cosine);
		device_array hankel_d(hankel);

		device_array res_d(resistivity);
		device_array height_d(height);
		device_array time_d(time);

		device_array b(time.size());
		device_array late_m_d(time.size());
		device_array late_e_d(time.size());

		forward_kernel << <time_d.size(), cosine_d.size(), sizeof(float_t) * cosine_d.size() >> >(
			a, i0, h,
			cosine_d.get_device_ptr(), hankel_d.get_device_ptr(),
			res_d.get_device_ptr(), height_d.get_device_ptr(),
			time_d.get_device_ptr(), b.get_device_ptr());
		auto err = cudaDeviceSynchronize();
		CHECK;

		calc_response_kernel << <1, time_d.size() >> >(
			a, i0, h,
			time_d.get_device_ptr(),
			b.get_device_ptr(),
			late_e_d.get_device_ptr(),
			late_m_d.get_device_ptr());
		err = cudaDeviceSynchronize();
		CHECK;

		late_m_d.save_data(response_late_m);
		late_e_d.save_data(response_late_e);
	}
}
