#pragma once

#include <cmath>
#include <cassert>
#include <sstream>
#include <vector>
#include <exception>

#include <cuda_runtime.h>

#include "../data/global.h"
#include "device_ptr.h"

//deviceÊý×é
class device_array
{
public:
	using float_t = global::float_t;

private:
	device_ptr data;

public:
	device_array() = default;
	device_array(const std::vector<float_t>& vec)
	{
		data.allocate(vec.size());
		copy_to_device()
		vec.data();
	}
};