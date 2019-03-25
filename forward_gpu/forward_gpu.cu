
#include <sstream>
#include <exception>

#include <cuda_runtime.h>

#include "forward_gpu.h"
#include "../data/data.h"
#include "../data/global.h"

//host中的数据
using float_h=global::float_t;
//device中的数据
using float_d=global::float_t;

__global__ void test_device()
{
	
}

#define CHECK_CUDA_ERROR(err)\
	if(err!=cudaSuccess)\
	{\
		std::stringstream msg;\
		msg << "cuda error: \n";\
		msg << cudaGetErrorName(err) << '\n';\
		msg << "at line: " << __LINE__;\
		throw std::runtime_error(msg.str());\
	}

void forward_gpu::init_cuda_device()
{
	int device_count;
	auto err = cudaGetDeviceCount(&device_count);
	CHECK_CUDA_ERROR(err);

}