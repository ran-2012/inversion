
#include "data_py.h"

namespace py = pybind11;

PYBIND11_MODULE(data_py, m)
{
	m.doc() = u"正反演程序中所需的基本数据结构";
	m.def("test_func", []() {return "hello world!"; });
	using float_t=float;
	//using float_t=double;
	//using float_t=long double;
	using geoelectric_model=geoelectric_model<float_t>;
	using forward_data=forward_data<float_t>;

	auto g=py::class_<geoelectric_model>(m, "geoelectric_model");
	g.def_readwrite("name", &geoelectric_model::name);
	g.def_readwrite("version", &geoelectric_model::version);
	g.def_readwrite("comment", &geoelectric_model::comment);
	g.def_readwrite("count", &geoelectric_model::count);

	g.def(py::init<>());
	g.def(py::init<geoelectric_model>());
	g.def("load_from_file", &geoelectric_model::load_from_file);
	g.def("save_to_file", &geoelectric_model::save_to_file);

	auto f = py::class_<forward_data>(m, "forward_data");
	f.def(py::init<forward_data>());


}



