#pragma once

#include <vector>

#include "../global/global.h"

namespace gpu
{
	using float_t = global::float_t;
	using vector = global::vector;

	void init_cuda_device();

	void test_cuda_device();

	void forward(float_t a, float_t i0, float_t h,
	             const vector& consine, const vector& hankel,
	             const vector& resistivity, const vector& height,
	             const vector& time,
	             vector& response_late_m,
	             vector& response_late_e);
}
