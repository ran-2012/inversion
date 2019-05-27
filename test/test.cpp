#pragma comment(lib, "global.lib")
#pragma comment(lib, "forward_gpu.lib")
#pragma comment(lib, "cudart_static.lib")

#include <string>
#include <iostream>

#include "../global/global.h"
#include "../data/data.h"
#include "../forward/forward.h"

//各种数据类型加载保存测试
void data_test()
{
	LOG("data_test start");
	TIMER();

	isometric_model g;
	geoelectric_model m;
	forward_data f;

	//加载数据
	g.load_from_file("../test_data/data_load_test.json");
	//保存数据
	g.save_to_file("../test_data/save_test.json");
	//等间距到不等间距模型可以进行转换
	m = g;
	//保存数据
	m.save_to_file(("../test_data/save_test2.json"));
	//生成对数时间
	f.generate_default_time_stamp();
	f.save_to_file("../test_data/save_test_forward.json");

	LOG("data_test end");
}

//cuda测试
void cuda_test()
{
	LOG("cuda_test start");
	TIMER();

	forward_gpu f;
	//利用预编好的cuda程序测试cuda设备
	f.test_cuda_device();

	LOG("cuda_test end");
}

//正演测试
void forward_test()
{
	LOG("forward_test start");
	TIMER();

	//正演类
	forward_gpu f;

	//滤波系数
	filter_coefficient coef;
	//地电模型
	geoelectric_model geo, geo2;
	//等间距地电模型
	isometric_model iso;
	//正演结果，此处用于给正演类添加时间数据
	forward_data data;

	//加载滤波系数
	coef.load_cos_coef("../test_data/cos_xs.txt");
	coef.load_hkl_coef("../test_data/hankel1.txt");
	//加载地电模型
	geo.load_from_file("../test_data/test_geo_model.json");
	iso.load_from_file("../test_data/test_iso_model.json");
	//等间距模型转换到不等间距模型
	geo2 = iso;
	//生成时间信息
	data.generate_time_stamp_by_count(-5, 0, 40);

	//正演类加载各种数据
	//加载等效半径、电流、高度信息
	f.load_general_params(10, 100, 50);
	//加载滤波系数
	f.load_filter_coef(coef);
	//加载地电模型
	f.load_geo_model(geo);
	//加载时间信息
	f.load_time_stamp(data);

	//正演开始
	f.forward();

	//获得结果
	//晚期磁场视电阻率
	auto data_late_m = f.get_result_late_m();
	//晚期感应电动势视电阻率
	auto data_late_e = f.get_result_late_e();

	auto n = data_late_m["idx"];
	auto t = data_late_m["time"];
	auto r = data_late_m["response"];

	for (size_t i = 0; i < n.size(); ++i)
	{
		std::cout << i << ' ' << t[i] << ' ' << r[i] << std::endl;
	}

	//密集计算测试
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
