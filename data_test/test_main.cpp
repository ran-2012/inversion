
#include <string>
#include <iostream>
#include <cstdlib>

#include "../data/data.h"
#include "../global/global.h"

int main()
{
	isometric_model<float> g;

	std::string tag("data test");

	global::log(tag, "begin data test");

	g.load_from_file("../data_load_test.json");
	g.save_to_file("../save_test.json");

	global::log(tag, "end data test");

	return 0;
}
