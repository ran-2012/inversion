#pragma once

#include <vector>
#include <iostream>
#include <sstream>
#include <exception>

#include "../data/data.h"
#include "../data/global.h"

class forward_base
{
public:
	using float_t=global::float_t;
	using string=std::string;
	using vector=std::vector<float_t>;
	using geoelectric_model=isometric_model<float_t>;
	using forward_data=forward_data<float_t>;
	using filter_coefficient=filter_coefficient<float_t>;

	constexpr static float_t threshold = 1e-5;

protected:
	filter_coefficient f;
	geoelectric_model g;
	forward_data d;

	virtual void check_coef()
	{
		if (f.hkl_coef.empty() || f.sin_coef.empty() || f.cos_coef.empty() || f.gs_coef.empty())
		{
			throw std::runtime_error("没有绑定滤波参数");
		}
	}

public:
	forward_base() = default;

	virtual ~forward_base() = default;

	virtual void bind_filter_coef(const filter_coefficient& coef) final
	{
		f = coef;
	}

	virtual void load_geo_model(geoelectric_model &mod)
	{
		g = mod;
	}

	virtual forward_data forward() = 0;
};

class forward_gpu :public forward_base
{
public:
	using forward_data = forward_base::forward_data;

protected:

public:
	forward_gpu() = default;
	~forward_gpu() = default;

	void init_cuda_device();
	void test_cuda_device();

	forward_data forward()override;
};