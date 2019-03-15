#pragma once

#include "data_model_base.h"

template<typename T>
class geoelectric_model :public data_model_base<T>
{
	using json=nlohmann::json;

	using string=std::string;
	using pair=std::pair<T, T>;
	using vector=std::vector<pair>;

	static constexpr char first_name[] = "number_of_layer";
	static constexpr char second_name[] = "resistivity";
	string _first_name() override { return first_name;}
	string _second_name() override { return second_name; }
	virtual string _layer_height() { return string("layer_height"); }


public:
	T layer_height;

	void load_from_json(const json& j) override
	{
		data_model_base<T>::load_from_json(j);

	}
};
