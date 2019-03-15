#pragma once

#include "data_model_base.h"

template<typename floatT>
class forward_data:public data_model_base<floatT>
{
	using string=std::string;
	static constexpr char first_name[] = "time";
	static constexpr char second_name[] = "response";
	string _first_name() override { return first_name; }
	string _second_name() override { return second_name; }


public:



};

