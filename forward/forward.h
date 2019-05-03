#pragma once

#include <vector>

#include "../data/data.h"
#include "../global/global.h"

class forward_base
{
public:
	using float_t = global::float_t;
	using string = std::string;
	using vector = global::vector;

	constexpr static float_t threshold = 1e-5;

protected:
	float_t a;
	float_t i0;
	float_t h;

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

	virtual ~forward_base() = default;

	forward_base& operator=(const forward_base& f) = default;

	virtual void load_general_param(const vector& v)
	{
		a = v[0];
		i0 = v[1];
		h = v[2];
	}
	virtual void load_filter_coef(const filter_coefficient& coef) final { filter = coef; }

	virtual void load_geo_model(const geoelectric_model& mod) final { geomodel = mod; }

	virtual void load_time_stamp(const forward_data& data) final { time_stamp = data; }

	virtual void forward() = 0;
};

class forward_gpu final : public forward_base
{
public:
	forward_gpu() = default;
	~forward_gpu() = default;

	forward_gpu& operator=(const forward_gpu& f) = default;

	static void init_cuda_device();
	static void test_cuda_device();

	void forward() override;
};
