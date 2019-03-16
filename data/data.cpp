
#include "data.h"

namespace py = pybind11;

PYBIND11_MODULE(geoelectric_data, m)
{
	auto g=py::class_<geoelectric_model<float>>(m, "geoelectric_model");
	g.def(py::init<geoelectric_model<float>&>());
	g.def("load_from_file", &geoelectric_model<float>::load_from_file);
	g.def("save_to_file", &geoelectric_model<float>::save_to_file);

}