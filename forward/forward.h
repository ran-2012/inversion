#pragma once

#include <vector>

#include "../data/data.h"
#include "../global/global.h"
#include "../forward_gpu/forward_gpu.h"

class forward_base
{
public:
	using float_t = global::float_t;
	using string = std::string;
	using vector = global::vector;

	constexpr static float_t threshold = 1e-5;

protected:
	float_t a = 10;
	float_t i0 = 1;
	float_t h = 0;

	filter_coefficient filter;
	geoelectric_model geomodel;

	forward_data time_stamp;
	forward_data data_late_e;
	forward_data data_late_m;

	virtual bool check_coef()
	{
		return !(filter.hkl_coef.empty() || filter.cos_coef.empty());
	}

public:
	forward_base() = default;

	forward_base(const forward_base& f) = default;
	forward_base(forward_base&& f) = default;

	virtual ~forward_base() = default;

	forward_base& operator=(const forward_base& f) = default;
	forward_base& operator=(forward_base&& f) = default;

	virtual void load_general_params(float_t a, float_t i0, float_t h)
	{
		this->a = a;
		this->i0 = i0;
		this->h = h;
	}

	virtual void load_general_params_s(float_t a, float_t i0, float_t h)
	{
		load_general_params(a, i0, h);
	}

	virtual void load_general_params(const vector& v) { load_general_params(v[0], v[1], v[2]); }

	virtual void load_filter_coef(const filter_coefficient& coef) final { filter = coef; }
	virtual void load_geo_model(const geoelectric_model& mod) final { geomodel = mod; }
	virtual void load_time_stamp(const forward_data& data) final { time_stamp = data; }

	virtual void forward() = 0;

	virtual forward_data get_result_late_m() { return data_late_m; }
	virtual forward_data get_result_late_e() { return data_late_e; }
};

class forward_gpu final : public forward_base
{
public:
	forward_gpu() = default;
	forward_gpu(const forward_gpu& f) = default;
	forward_gpu(forward_gpu&& f) = default;

	~forward_gpu() = default;

	forward_gpu& operator=(const forward_gpu& f) = default;
	forward_gpu& operator=(forward_gpu&& f) = default;

	/**
	 * \brief 初始化CUDA设备
	 */
	void init_cuda_device()
	{
		try
		{
			gpu::init_cuda_device();
		}
		catch (std::exception& e)
		{
			global::err(e.what());
			global::err("init cuda device failed");
		}
		LOG("init cuda device completed");
	}

	/**
	 * \brief 测试CUDA设备
	 */
	void test_cuda_device()
	{
		try
		{
			gpu::test_cuda_device();
		}
		catch (std::exception& e)
		{
			global::err(e.what());
			global::err("test cuda device failed");
		}
	}

	/**
	 * \brief 正演
	 */
	void forward() override
	{
		assert(geomodel.size());
		assert(check_coef());

		if (time_stamp.size() == 0)
		{
			time_stamp.generate_default_time_stamp();
		}

		data_late_m = time_stamp;
		data_late_e = time_stamp;

		try
		{
			gpu::forward(a, i0, h,
			             filter.get_cos(), filter.get_hkl(),
			             geomodel["resistivity"], geomodel["height"],
			             time_stamp["time"],
			             data_late_m["response"], data_late_e["response"]);

			//电场响应的最后一组数据是无效的
			data_late_e.pop_back();
		}
		catch (std::exception& e)
		{
			global::err(e.what());
			global::err("forward failed");
		}
	}
};
