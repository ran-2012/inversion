
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "data.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<global::float_t>);

PYBIND11_MODULE(data_py, m)
{
	m.doc() = "data and model in forward process";

	py::bind_vector<std::vector<global::float_t>>(m, "vector_float_t");

	m.def("test_func", []() {return "hello world!"; }, "test function");
	m.def("vec_test_func", []() {return std::vector<global::float_t>{1, 2, 3}; });

	//data_model_base
	auto b = py::class_<data_model_base>(m, "data_model_base");
	b.doc() = "data model base class";
	b.def_readwrite("name", &data_model_base::name ,"name");
	b.def_readwrite("version", &data_model_base::version, "version");
	b.def_readwrite("comment", &data_model_base::comment, "content description");

	b.def("__getitem__", &data_model_base::get_item_s);
	b.def("__setitem__", &data_model_base::set_item);
	b.def("get_content_name", &data_model_base::get_content_name);
	b.def_property_readonly("count", &data_model_base::size, "number of data");

	b.def(py::init<>());
	b.def(py::init<data_model_base>());
	b.def("load_from_file", &data_model_base::load_from_file);
	b.def("save_to_file", &data_model_base::save_to_file);

	//geoelectric_model
	auto g = py::class_<geoelectric_model>(m, "geoelectric_model", b);
	g.doc() = "general geoelectric model";

	g.def(py::init<>());

	//isometric_model
	auto i = py::class_<isometric_model>(m, "isometric_model", b);
	i.doc() = "isometric model";
	i.def_readwrite("layer_height", &isometric_model::layer_height);

	i.def(py::init<>());

	//forward_data
	auto f = py::class_<forward_data>(m, "forward_data", b);
	f.doc() = "forward data";

	f.def(py::init<>());

	//filter_coefficient
	auto c = py::class_<filter_coefficient>(m, "filter_coefficient");
	c.doc() = "filter coefficient";

	c.def(py::init<>());
	c.def(py::init<filter_coefficient>());
	c.def("load_hkl_coef", &filter_coefficient::load_hkl_coef);
	c.def("load_sin_coef", &filter_coefficient::load_sin_coef);
	c.def("load_cos_coef", &filter_coefficient::load_cos_coef);
	c.def("load_gs_coef", &filter_coefficient::load_gs_coef);

}


