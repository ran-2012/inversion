
#include <iostream>
#include <cstdlib>

#include "../data/geoelectric_model.h"

int main()
{
#ifdef _WIN64
	system("chcp 65001");
#endif

	geoelectric_model<float> g;
	g.load_from_file("../data_load_test.json");

	return 0;
}
