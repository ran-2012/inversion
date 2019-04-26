#pragma comment(lib, "global.lib")
#pragma comment(lib, "forward_gpu.lib")
#pragma comment(lib, "forward.lib")
#pragma comment(lib, "cudart_static.lib")

#include <iostream>
#include <exception>

#include "../global/global.h"
#include "../data/data.h"
#include "../forward_gpu/forward_gpu.h"
#include "../forward/forward.h"

int main()
{
	int i = 0;
	++i;
	global::scoped_timer timer("test");

	forward_gpu f;
	forward_gpu::test_cuda_device();
	// filter_coefficient<global::float_t> coef;
	// geoelectric_model<global::float_t> geo;
	//
	// coef.load_cos_coef("cos.txt");
	// coef.load_hkl_coef("hkl.txt");
	//
	// geo.load_from_file("geo.json");
	//
	// f.load_filter_coef(coef);
	// f.load_geo_model(geo);

	return 0;
}
