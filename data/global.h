#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <exception>

class global
{
	using vector=std::vector<double>;
	using string=std::string;

public:
	static constexpr double pi = 3.14159265359;
	static constexpr double mu0 = 1.25663706e-6;
};
