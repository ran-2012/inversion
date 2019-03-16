#pragma once

#include "data_model_base.h"

template<typename FloatT>
class geoelectric_model :public data_model_base<FloatT>
{
	using json=nlohmann::json;

	using string=std::string;
	using pair=std::pair<FloatT, FloatT>;
	using vector=std::vector<pair>;

	static constexpr char first_name[] = "number_of_layer";
	static constexpr char second_name[] = "resistivity";
	string _first_name() override { return first_name;}
	string _second_name() override { return second_name; }
	virtual string _layer_height() { return string("layer_height"); }
	
	void load_additional_data(const json& j) override
	{
		layer_height = j[_layer_height()].get<FloatT>();
	}
	json save_additional_data() override
	{
		json patch_j;
		patch_j[_layer_height()] = layer_height;
		return patch_j;
	}

public:
	FloatT layer_height;

	geoelectric_model(const geoelectric_model<FloatT>& g) :data_model_base<FloatT>(g)
	{
		layer_height = g.layer_height;
	}
	geoelectric_model(geoelectric_model<FloatT>&& g)noexcept :data_model_base<FloatT>(std::move(g))
	{
		layer_height = std::move(g.layer_height);
	}

	geoelectric_model<FloatT>& operator=(const geoelectric_model<FloatT>& g)
	{
		*this = data_model_base<FloatT>::operator=(g);
		layer_height = g.layer_height;
		return *this;
	}
	geoelectric_model<FloatT>& operator=(geoelectric_model<FloatT>&& g) noexcept
	{
		*this = data_model_base<FloatT>::operator=(std::move(g));
		layer_height = std::move(g.layer_height);
		return *this;
	}
};
