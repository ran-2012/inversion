#pragma once

#include "data_model_base.h"

template<typename FloatT = float>
class forward_data :public data_model_base<FloatT>
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

	forward_data() :data_model_base<FloatT>()
	{

	}
	forward_data(const forward_data<FloatT>& f) :data_model_base<FloatT>(f)
	{

	}
	forward_data(forward_data<FloatT>&& f)noexcept :data_model_base<FloatT>(std::move(f))
	{

	}
	virtual ~forward_data() = default;

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

template<typename FloatT = float>
class geoelectric_model :public data_model_base<FloatT>
{
protected:
	using json=nlohmann::json;

	using string=std::string;
	using pair=std::pair<FloatT, FloatT>;
	using vector=std::vector<pair>;

	static constexpr char first_name[] = "number_of_layer";
	static constexpr char second_name[] = "resistivity";

	virtual void load_layer_height(const json& j)
	{
		if (!j.count(_layer_height()))
		{
			data_model_base<FloatT>::
				throw_critical_data_miss_exception(_layer_height());
		}
		layer_height = j[_layer_height()].get<FloatT>();
	}

	string _first_name() override { return first_name; }
	string _second_name() override { return second_name; }
	virtual string _layer_height() { return string("layer_height"); }

	void load_additional_data(const json& j) override
	{
		load_layer_height(j);
	}
	json save_additional_data() override
	{
		json patch_j;
		patch_j[_layer_height()] = layer_height;
		return patch_j;
	}

public:
	FloatT layer_height;

	geoelectric_model() :data_model_base<FloatT>()
	{
		layer_height = 0;
	}
	geoelectric_model(const geoelectric_model<FloatT>& g) :data_model_base<FloatT>(g)
	{
		layer_height = g.layer_height;
	}
	geoelectric_model(geoelectric_model<FloatT>&& g)noexcept :data_model_base<FloatT>(std::move(g))
	{
		layer_height = std::move(g.layer_height);
	}
	virtual ~geoelectric_model() = default;

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

class filter_coefficient
{
public:
	using vector=std::vector<double>;
	using string=std::string;

	vector hkl_idx;
	vector sin_idx;
	vector cos_idx;
	vector gs_idx;

	void load_idx_from_file(const string& path, vector& v)
	{
		std::ifstream input_file;
		try
		{
			input_file.open(path, std::ifstream::in);
			if (!input_file)
			{
				throw std::runtime_error(path + u8"´ò¿ªÊ§°Ü");
			}
		}
		catch (std::exception & e)
		{
			std::cerr << e.what() << std::endl;
			return;
		}

		int n;
		input_file >> n;
		v.clear();
		while (n--)
		{
			double idx;
			input_file >> idx;
			v.emplace_back(idx);
		}
	}
	void load_hkl_coef(const string & path)
	{
		load_idx_from_file(path, hkl_idx);
	}
	void load_sin_coef(const string & path)
	{
		load_idx_from_file(path, sin_idx);
	}
	void load_cos_coef(const string & path)
	{
		load_idx_from_file(path, cos_idx);
	}
	void load_gs_coef(const string & path)
	{
		load_idx_from_file(path, gs_idx);
	}

};