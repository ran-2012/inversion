#include "../forward_gpu/forward_gpu.h"
#include "forward.h"
#include <ctime>

void forward_gpu::init_cuda_device()
{
	gpu::init_cuda_device();
}

void forward_gpu::test_cuda_device()
{
	gpu::test_cuda_device();
}

void forward_gpu::forward()
{
	assert(geomodel.size());
	assert(check_coef());
	if (time_stamp.size() == 0)
	{
		time_stamp.generate_default_time_stamp();
	}

	data_late_e = time_stamp;
	data_late_m = time_stamp;

	gpu::forward(a, i0, h,
	             filter.get_cos(), filter.get_hkl(),
	             geomodel["resistivity"], geomodel["height"],
	             time_stamp["time"],
	             data_late_e["response"], data_late_m["response"]);
}
