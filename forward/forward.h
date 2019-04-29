#pragma once

#include <vector>
#include <iostream>
#include <sstream>
#include <exception>

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
	filter_coefficient filter;
	geoelectric_model geomodel;

	forward_data time_stamp;
	forward_data data;

	virtual bool check_coef()
	{
		return !(filter.hkl_coef.empty() || filter.cos_coef.empty());
	}

public:
	forward_base() = default;

	virtual ~forward_base() = default;

	virtual void load_filter_coef(const filter_coefficient& coef) final { filter = coef; }

	virtual void load_geo_model(const geoelectric_model& mod) final { geomodel = mod; }

	virtual void load_forward_data(const forward_data& data) final { time_stamp = data; }

	virtual forward_data forward() = 0;
};

class forward_gpu final : public forward_base
{
public:
	forward_gpu() = default;
	~forward_gpu() = default;

	static void init_cuda_device();
	static void test_cuda_device();

	forward_data forward() override;
};
