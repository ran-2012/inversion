//this file is encoded in ASCII due to python may not recognize other encoding correctly
#pragma comment(lib, "global.lib")
#pragma comment(lib, "forward_gpu.lib")
#pragma comment(lib, "cudart_static.lib")

#include <exception>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "../data/data.h"
#include "../forward/forward.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<global::float_t>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<global::float_t>>);

PYBIND11_MODULE(data_py, m)
{
	m.doc() = "data and model in forward process";

	py::bind_vector<std::vector<global::float_t>>(m, "vector_float_t");
	py::bind_vector<std::vector<std::vector<global::float_t>>>(m, "vector_vector_float_t");

	m.def("test_func", []() { return "hello world!"; }, "test function");
	m.def("vec_test_func", []() { return std::vector<global::float_t>{1, 2, 3}; });
	m.def("iso_to_geo", [](const isometric_model& i)
	{
		geoelectric_model g;
		g = i;
		return g;
	});

	//data_model_base
	auto b = py::class_<data_model_base>(m, "data_model_base");
	b.doc() = "data model base class";
	b.def_readwrite("name", &data_model_base::name, "name");
	b.def_readwrite("version", &data_model_base::version, "version");
	b.def_readwrite("comment", &data_model_base::comment, "content description");
	b.def_readwrite("data", &data_model_base::data);

	b.def("__getitem__", &data_model_base::get_item_s);
	b.def("__setitem__", &data_model_base::set_item_s);
	b.def("get_content_name", &data_model_base::get_content_name);
	b.def_property_readonly("count", &data_model_base::size, "number of data");

	b.def(py::init<>());
	b.def(py::init<data_model_base>());
	b.def("resize", &data_model_base::resize);
	b.def("load_from_file", &data_model_base::load_from_file, "load data from json file");
	b.def("save_to_file", &data_model_base::save_to_file, "save data to json file");

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
	f.def("generate_time_stamp_by_count", &forward_data::generate_time_stamp_by_count);
	f.def("generate_default_time_stamp", &forward_data::generate_default_time_stamp);

	//filter_coefficient
	auto c = py::class_<filter_coefficient>(m, "filter_coefficient");
	c.doc() = "filter coefficient";

	c.def(py::init<>());
	c.def(py::init<filter_coefficient>());
	c.def("load_hkl_coef", &filter_coefficient::load_hkl_coef);
	c.def("load_sin_coef", &filter_coefficient::load_sin_coef);
	c.def("load_cos_coef", &filter_coefficient::load_cos_coef);
	c.def("load_gs_coef", &filter_coefficient::load_gs_coef);

	auto fw = py::class_<forward_gpu>(m, "forward_gpu");
	fw.doc() = "GPU forward";

	fw.def(py::init<>());
	fw.def_readwrite("a", &forward_gpu::a);
	fw.def_readwrite("i0", &forward_gpu::i0);
	fw.def_readwrite("h", &forward_gpu::h);

	fw.def_readwrite("filter", &forward_gpu::filter);
	fw.def_readwrite("geo_model", &forward_gpu::geomodel);
	fw.def_readwrite("time_stamp", &forward_gpu::time_stamp);
	fw.def_readwrite("magnetic", &forward_gpu::magnetic);
	fw.def_readwrite("a_resistivity_late_e", &forward_gpu::a_resistivity_late_e);
	fw.def_readwrite("a_resistivity_late_m", &forward_gpu::a_resistivity_late_m);

	fw.def("load_general_params", &forward_gpu::load_general_params_s);
	fw.def("load_geo_model", &forward_gpu::load_geo_model);
	fw.def("load_filter_coef", &forward_gpu::load_filter_coef);
	fw.def("load_time_stamp", &forward_gpu::load_time_stamp);

	fw.def("init_cuda_device", &forward_gpu::init_cuda_device);
	fw.def("test_cuda_device", &forward_gpu::test_cuda_device);

	fw.def("forward", &forward_gpu::forward);
	fw.def("gradient", &forward_gpu::gradient);

	fw.def("get_result_magnetic", &forward_gpu::get_result_magnetic);
	fw.def("get_result_late_m", &forward_gpu::get_result_late_m);
	fw.def("get_result_late_e", &forward_gpu::get_result_late_e);
}
