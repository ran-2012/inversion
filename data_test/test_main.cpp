
#include <iostream>
#include <cstdlib>

#include "../data/data.h"

int main()
{
	isometric_model<float> g;
	g.load_from_file("../data_load_test.json");
	g.save_to_file("../save_test.json");
	return 0;
}
