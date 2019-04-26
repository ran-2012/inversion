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
	return forward_data();
}
