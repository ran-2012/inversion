#pragma comment(lib, "global.lib")
#pragma comment(lib, "forward_gpu.lib")
#pragma comment(lib, "forward.lib")
#pragma comment(lib, "cudart_static.lib")

#include <iostream>
#include <exception>

#include <string>
#include <iostream>
#include <cstdlib>

#include "../global/global.h"
#include "../data/data.h"
#include "../forward_gpu/forward_gpu.h"
#include "../forward/forward.h"

void data_test()
{
	LOG("data_test start");

	isometric_model g;

	std::string tag("data test");

	global::log(tag, "begin data test");

	g.load_from_file("../data_load_test.json");
	g.save_to_file("../save_test.json");

	global::log(tag, "end data test");

	LOG("data_test end");
}

void cuda_test()
{
	LOG("cuda_test start");

	global::scoped_timer timer("test");

	forward_gpu f;
	forward_gpu::test_cuda_device();

	LOG("cuda_test end");
}

void forward_test()
{
	LOG("forward_test start");

	global::scoped_timer timer("forward_test");

	filter_coefficient coef;
	geoelectric_model geo;

	coef.load_cos_coef("cos.txt");
	coef.load_hkl_coef("hkl.txt");

	geo.load_from_file("geo.json");

	LOG("forward_test end");
}

int main()
{
	data_test();


	return 0;
}
