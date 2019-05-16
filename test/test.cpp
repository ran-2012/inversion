#pragma comment(lib, "global.lib")
#pragma comment(lib, "forward_gpu.lib")
#pragma comment(lib, "cudart_static.lib")

#include <string>
#include <iostream>

#include "../global/global.h"
#include "../data/data.h"
#include "../forward/forward.h"

void data_test()
{
	LOG("data_test start");
	TIMER();

	isometric_model g;
	geoelectric_model m;
	forward_data f;

	std::string tag("data test");

	g.load_from_file("../test_data/data_load_test.json");
	g.save_to_file("../test_data/save_test.json");
	m = g;
	m.save_to_file(("../test_data/save_test2.json"));
	f.generate_default_time_stamp();
	f.save_to_file("../test_data/save_test_forward.json");

	LOG("data_test end");
}

void cuda_test()
{
	LOG("cuda_test start");
	TIMER();

	forward_gpu f;
	f.test_cuda_device();

	LOG("cuda_test end");
}

void forward_test()
{
	LOG("forward_test start");
	TIMER();

	forward_gpu f;

	filter_coefficient coef;
	geoelectric_model geo, geo2;
	isometric_model iso;
	forward_data data;

	coef.load_cos_coef("../test_data/cos_xs.txt");
	coef.load_hkl_coef("../test_data/hankel1.txt");
	geo.load_from_file("../test_data/test_geo_model.json");
	iso.load_from_file("../test_data/test_iso_model.json");
	geo2 = iso;
	data.generate_time_stamp_by_count(-5, 0, 40);

	f.load_general_params(10, 100, 50);
	f.load_filter_coef(coef);
	f.load_geo_model(geo);
	f.load_time_stamp(data);

	f.forward();

	auto data_late_m = f.get_result_late_m();
	auto data_late_e = f.get_result_late_e();

	auto n = data_late_m["idx"];
	auto t = data_late_m["time"];
	auto r = data_late_m["response"];

	for (size_t i = 0; i < n.size(); ++i)
	{
		std::cout << i << ' ' << t[i] << ' ' << r[i] << std::endl;
	}

	LOG("serieal");
	{
		TIMER();
		f.load_geo_model(geo2);

		for (size_t i = 0; i < geo2.size(); ++i)
		{
			geo2["resistivity"][i] += 10;
			f.load_geo_model(geo2);
			f.forward();
		}
	}

	LOG("parellal");
	f.load_geo_model(geo2);
	auto grads = f.gradient(10);

	LOG("forward_test end");
}

void clock_test()
{
}

int main() noexcept
{
	try
	{
		data_test();
		cuda_test();
		forward_test();
	}
	catch (...)
	{
		LOG("test failed");
	}

	return 0;
}
