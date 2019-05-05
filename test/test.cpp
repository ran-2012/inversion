#pragma comment(lib, "global.lib")
#pragma comment(lib, "forward_gpu.lib")
#pragma comment(lib, "cudart_static.lib")

#include <iostream>
#include <exception>

#include <string>
#include <iostream>
#include <cstdlib>

#include "../global/global.h"
#include "../data/data.h"
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

	forward_gpu f;

	filter_coefficient coef;
	geoelectric_model geo;
	forward_data data;

	coef.load_cos_coef("cos.txt");
	coef.load_hkl_coef("hkl.txt");
	geo.load_from_file("geo.json");
	data.generate_time_stamp(-6, 0, 40);

	f.load_general_params(10, 1, 2);
	f.load_filter_coef(coef);
	f.load_geo_model(geo);
	f.load_time_stamp(data);

	f.forward();

	auto data_late_m = f.get_result_late_m();
	auto data_late_e = f.get_result_late_e();

	auto n = data_late_m["idx"];
	auto t = data_late_m["time"];
	auto r = data_late_m["response"];

	for (auto i = 0; i < n.size(); ++i)
	{
		std::cout << i << ' ' << t[i] << ' ' << r[i] << std::endl;
	}

	LOG("forward_test end");
}

int main()
{
	data_test();
	cuda_test();

	return 0;
}
