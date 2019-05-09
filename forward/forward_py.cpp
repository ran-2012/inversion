
//this file is encoded in ASCII due to python may not recognize other encoding correctly
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
	m.doc() = "forward module";

	auto fw = py::class_<forward_gpu>(m, "forward_gpu");
	fw.doc() = "GPU forward";

	fw.def(py::init<>());
	fw.def("load_general_params", &forward_gpu::load_general_params_s);
	fw.def("load_geo_model", &forward_gpu::load_geo_model);
	fw.def("load_filter_coef", &forward_gpu::load_filter_coef);
	fw.def("load_time_stamp", &forward_gpu::load_time_stamp);

	fw.def("init_cuda_device", &forward_gpu::init_cuda_device);
	fw.def("test_cuda_device", &forward_gpu::test_cuda_device);

	fw.def("forward", &forward_gpu::forward);

	fw.def("get_result_late_m", &forward_gpu::get_result_late_m);
	fw.def("get_result_late_e", &forward_gpu::get_result_late_e);
}
