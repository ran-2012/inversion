
#include <pybind11/pybind11.h>

#include "data.h"
#include "global.h"

namespace py = pybind11;

static global g;

PYBIND11_MODULE(data_py, m)
{
	m.doc() = u8"正反演程序中所需的基本数据结构及全局数据";
	m.def("test_func", []() {return "hello world!"; }, u8"测试函数");
	using float_t=float;
	using geoelectric_model=geoelectric_model<float_t>;
	using forward_data=forward_data<float_t>;

	auto g=py::class_<geoelectric_model>(m, "geoelectric_model");
	g.doc() = u8"地电模型";
	g.def_readwrite("name", &geoelectric_model::name);
	g.def_readwrite("version", &geoelectric_model::version);
	g.def_readwrite("comment", &geoelectric_model::comment);
	g.def("__getitem__", &geoelectric_model::operator[]);
	g.def("__setitem__", &geoelectric_model::set_item);
	g.def_property_readonly("count", &geoelectric_model::size);

	g.def(py::init<>());
	g.def(py::init<geoelectric_model>());
	g.def("load_from_file", &geoelectric_model::load_from_file);
	g.def("save_to_file", &geoelectric_model::save_to_file);

	auto f = py::class_<forward_data>(m, "forward_data");
	f.doc() = u8"正演数据";
	f.def(py::init<forward_data>());

	auto c = py::class_<filter_coefficient>(m, "filter_coefficient");
	c.def("load_hkl_coef", &filter_coefficient::load_hkl_coef);
	c.def("load_sin_coef", &filter_coefficient::load_sin_coef);
	c.def("load_cos_coef", &filter_coefficient::load_cos_coef);
	c.def("load_gs_coef", &filter_coefficient::load_gs_coef);

}



