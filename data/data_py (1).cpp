
#include <pybind11/pybind11.h>

#include "data.h"

namespace py = pybind11;

PYBIND11_MODULE(data_py, m)
{
	m.doc() = "正反演程序中所需的基本数据结构及全局数据";
	m.def("test_func", []() {return "hello world!"; }, "测试函数");
	using float_t=float;
	using data_model_base=data_model_base<float_t>;
	using geoelectric_model=geoelectric_model<float_t>;
	using forward_data=forward_data<float_t>;

	//data_model_base
	auto b=py::class_<data_model_base>(m, "data_model_base");
	b.doc() = "数据模型基类";
	b.def_readwrite("name", &data_model_base::name);
	b.def_readwrite("version", &data_model_base::version);
	b.def_readwrite("comment", &data_model_base::comment);
	b.def("__getitem__", &data_model_base::operator[]);
	b.def("__setitem__", &data_model_base::set_item);
	b.def_property_readonly("count", &data_model_base::size);

	b.def(py::init<>());
	b.def(py::init<data_model_base>());
	b.def("load_from_file", &data_model_base::load_from_file);
	b.def("save_to_file", &data_model_base::save_to_file);

	//geoelectric_model
	auto g = py::class_<geoelectric_model>(m, "geoelectric_model", b);
	g.doc() = "地电模型";
	g.def_readwrite("layer_height", &geoelectric_model::layer_height);

	g.def(py::init<>());

	//forward_data
	auto f = py::class_<forward_data>(m, "forward_data", b);
	f.doc() = "正演数据";

	f.def(py::init<>());

	//filter_coefficient
	auto c = py::class_<filter_coefficient>(m, "filter_coefficient");
	c.def("load_hkl_coef", &filter_coefficient::load_hkl_coef);
	c.def("load_sin_coef", &filter_coefficient::load_sin_coef);
	c.def("load_cos_coef", &filter_coefficient::load_cos_coef);
	c.def("load_gs_coef", &filter_coefficient::load_gs_coef);

}



