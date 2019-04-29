#include "../forward_gpu/forward_gpu.h"
#include "forward.h"

void forward_gpu::init_cuda_device()
{
	gpu::init_cuda_device();
}

void forward_gpu::test_cuda_device()
{
	gpu::test_cuda_device();
}

forward_data forward_gpu::forward()
{
	assert(geomodel.size());
	assert(check_coef());
	if(data.size()==0)
	{
		data.generate_default_time_stamp();
	}

	gpu::forward(filter.get_cos(), filter.get_hkl(),
		geomodel["resistivity"], geomodel["height"],
		time_stamp["time"],
		data["response"], data["response"]);
	
	return forward_data();
}
