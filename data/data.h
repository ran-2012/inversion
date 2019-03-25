#pragma once

#include "data_model_base.h"

template<typename FloatT = global::float_t>
class forward_data :public data_model_base<FloatT>
{
	using string=std::string;
	using json=nlohmann::json;
	using size_type=typename data_model_base<FloatT>::size_type;

	static constexpr char first_name[] = "time";
	static constexpr char second_name[] = "response";

	std::vector<string> _data_content_name() override { return { first_name, second_name }; }

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
		this->data_model_base<FloatT>::operator=(f);

		return *this;
	}
	forward_data<FloatT>& operator=(forward_data<FloatT>&& f)noexcept
	{
		this->data_model_base<FloatT>::operator=(std::move(f));

		return *this;
	}
};

template<typename FloatT = global::float_t>
class geoelectric_model :public data_model_base<FloatT>
{
	using string=std::string;
	using json=nlohmann::json;
	using size_type=typename data_model_base<FloatT>::size_type;

	static constexpr char first_name[] = "idx";
	static constexpr char second_name[] = "height";
	static constexpr char third_name[] = "resistivity";

	std::vector<string> _data_content_name() override { return { first_name, second_name, third_name }; }

	void load_additional_data(const json& j) override
	{

	}
	json save_additional_data() override
	{
		json patch_j;
		return patch_j;
	}

public:

	geoelectric_model() :data_model_base<FloatT>()
	{

	}
	geoelectric_model(const geoelectric_model<FloatT>& f) :data_model_base<FloatT>(f)
	{

	}
	geoelectric_model(geoelectric_model<FloatT>&& f)noexcept :data_model_base<FloatT>(std::move(f))
	{

	}
	virtual ~geoelectric_model() = default;

	geoelectric_model<FloatT>& operator=(const geoelectric_model<FloatT>& f)
	{
		this->data_model_base<FloatT>::operator=(f);

		return *this;
	}
	geoelectric_model<FloatT>& operator=(geoelectric_model<FloatT>&& f)noexcept
	{
		this->data_model_base<FloatT>::operator=(std::move(f));

		return *this;
	}
};

template<typename FloatT = global::float_t>
class isometric_model :public data_model_base<FloatT>
{
protected:
	using json=nlohmann::json;
	using string=std::string;
	using pair=std::pair<FloatT, FloatT>;
	using size_type=typename data_model_base<FloatT>::size_type;

	static constexpr char first_name[] = "idx";
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

	std::vector<string> _data_content_name() override { return { first_name, second_name }; }
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

	isometric_model() :data_model_base<FloatT>()
	{
		layer_height = 0;
	}
	isometric_model(const isometric_model<FloatT>& g) :data_model_base<FloatT>(g)
	{
		layer_height = g.layer_height;
	}
	isometric_model(isometric_model<FloatT>&& g)noexcept :data_model_base<FloatT>(std::move(g))
	{
		layer_height = std::move(g.layer_height);
	}
	virtual ~isometric_model() = default;

	isometric_model<FloatT>& operator=(const isometric_model<FloatT>& g)
	{
		this->data_model_base<FloatT>::operator=(g);
		layer_height = g.layer_height;
		return *this;
	}
	isometric_model<FloatT>& operator=(isometric_model<FloatT>&& g) noexcept
	{
		this->data_model_base<FloatT>::operator=(std::move(g));
		layer_height = std::move(g.layer_height);
		return *this;
	}
};

template<typename FloatT = global::float_t>
class filter_coefficient
{
public:
	using vector=std::vector<FloatT>;
	using string=std::string;

	vector hkl_coef;
	vector sin_coef;
	vector cos_coef;
	vector gs_coef;

	filter_coefficient() = default;
	filter_coefficient(const filter_coefficient<FloatT>& coef)
	{
		*this = coef;
	}
	~filter_coefficient() = default;

	filter_coefficient<FloatT>& operator=(const filter_coefficient<FloatT>& coef)
	{
		hkl_coef = coef.hkl_coef;
		sin_coef = coef.sin_coef;
		cos_coef = coef.cos_coef;
		gs_coef = coef.gs_coef;

		return *this;
	}

	static void load_coef_from_file(const string& path, vector& v)
	{
		std::ifstream input_file;
		try
		{
			input_file.open(path, std::ifstream::in);
			if (!input_file)
			{
				throw std::runtime_error(string("file ") + path + string(" do not exist"));
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
			FloatT idx;
			input_file >> idx;
			v.emplace_back(idx);
		}
	}
	void load_hkl_coef(const string & path)
	{
		load_coef_from_file(path, hkl_coef);
	}
	void load_sin_coef(const string & path)
	{
		load_coef_from_file(path, sin_coef);
	}
	void load_cos_coef(const string & path)
	{
		load_coef_from_file(path, cos_coef);
	}
	void load_gs_coef(const string & path)
	{
		load_coef_from_file(path, gs_coef);
	}

};
