#pragma once

#include <vector>

#include "../global/global.h"

namespace gpu
{
	using vector = std::vector<global::float_t>;

	void init_cuda_device();

	void test_cuda_device();

	void forward(const vector& consine, const vector& hankel,
	             const vector& resistivity, const vector& height,
	             const vector& time,
	             vector& response_late_m,
	             vector& response_late_e);
}
