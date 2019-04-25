
#include <iostream>
#include <exception>

#include "../global/global.h"
#include "../forward_gpu/forward_gpu.h"

int main()
{
	global::scoped_timer timer("test");

	forward_gpu f;
	filter_coefficient<global::float_t> coef;
	geoelectric_model<global::float_t> geo;

	coef.load_cos_coef("cos.txt");
	coef.load_hkl_coef("hkl.txt");
	
	geo.load_from_file("geo.json");

	f.load_filter_coef(coef);
	f.load_geo_model(geo);

	return 0;
}