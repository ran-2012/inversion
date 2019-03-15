#pragma once

#include "data_model_base.h"

template<typename floatT>
class geoelectric_model :public data_model_base<floatT>
{
	using json=nlohmann::json;

	using string=std::string;
	using pair=std::pair<floatT, floatT>;
	using vector=std::vector<pair>;

	static constexpr char first_name[] = "number_of_layer";
	static constexpr char second_name[] = "resistivity";
	string _first_name() override { return first_name;}
	string _second_name() override { return second_name; }
	virtual string _layer_height() { return string("layer_height"); }
	
	void load_additional_data(const json& j) override
	{
		layer_height = j[_layer_height()].get<string>();
	}
	json save_additional_data() override
	{
		json patch_j;
		patch_j[_layer_height()] = layer_height;
		return patch_j;
	}

public:
	floatT layer_height;

};
