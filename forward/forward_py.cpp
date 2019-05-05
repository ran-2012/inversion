#pragma comment(lib, "global.lib")
#pragma comment(lib, "forward_gpu.lib")
#pragma comment(lib, "cudart_static.lib")

#include <exception>

#include <pybind11/pybind11.h>

#include "../data/data.h"
#include "forward.h"

namespace py = pybind11;

PYBIND11_MODULE(forward_py, m)
{
	m.doc() = "正演模块";

	auto f = py::class_<forward_gpu>(m, "forward_gpu");
	f.doc() = "GPU正演";

	f.def(py::init<>());
	f.def("load_geo_model", &forward_gpu::load_geo_model);
	f.def("load_filter_coef", &forward_gpu::load_filter_coef);
	f.def("load_time_stamp", &forward_gpu::load_time_stamp);

	f.def("forward", &forward_gpu::forward);
}
