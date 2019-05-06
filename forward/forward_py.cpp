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
	f.def("load_general_params", &forward_gpu::load_general_params_s);
	f.def("load_geo_model", &forward_gpu::load_geo_model);
	f.def("load_filter_coef", &forward_gpu::load_filter_coef);
	f.def("load_time_stamp", &forward_gpu::load_time_stamp);

	f.def("init_cuda_device", &forward_gpu::init_cuda_device);
	f.def("test_cuda_device", &forward_gpu::test_cuda_device);

	f.def("forward", &forward_gpu::forward);

	f.def("get_result_late_m", &forward_gpu::get_result_late_m);
	f.def("get_result_late_e", &forward_gpu::get_result_late_e);
}
