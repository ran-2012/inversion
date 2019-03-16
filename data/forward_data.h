#pragma once

#include "data_model_base.h"

template<typename FloatT>
class forward_data:public data_model_base<FloatT>
{
	using string=std::string;
	using json=nlohmann::json;

	static constexpr char first_name[] = "time";
	static constexpr char second_name[] = "response";
	string _first_name() override { return first_name; }
	string _second_name() override { return second_name; }
	void load_additional_data(const json& j) override
	{
		
	}
	json save_additional_data() override
	{
		json patch_j;
		return patch_j;
	}
public:

	forward_data(const forward_data<FloatT>& f):data_model_base<FloatT>(f)
	{
		
	}
	forward_data(forward_data<FloatT>&& f)noexcept :data_model_base<FloatT>(std::move(f))
	{
		
	}

	forward_data<FloatT>& operator=(const forward_data<FloatT>& f)
	{
		*this = data_model_base<FloatT>::operator=(f);

		return *this;
	}
	forward_data<FloatT>& operator=(forward_data<FloatT>&& f)noexcept
	{
		*this = data_model_base<FloatT>::operator=(std::move(f));

		return *this;
	}
};

