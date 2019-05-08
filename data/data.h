#pragma once

#include "data_model_base.h"

//正演数据
class forward_data;
//地电模型
class geoelectric_model;
//等厚地电模型
class isometric_model;

class forward_data final : public data_model_base
{
	static constexpr char second_name[] = "time";
	static constexpr char third_name[] = "response";

	std::vector<string> _data_content_name() override { return {index_name, second_name, third_name}; }

	void load_additional_data(const json& j) override
	{
	}

	json save_additional_data() override
	{
		json patch_j;
		return patch_j;
	}

public:

	forward_data() : data_model_base()
	{
	}

	forward_data(const forward_data& f) : data_model_base(f)
	{
	}

	forward_data(forward_data&& f) noexcept : data_model_base(std::move(f))
	{
	}

	virtual ~forward_data() = default;

	size_type get_name_idx(const string& name) const override
	{
		if (name == string(index_name))
			return 0;
		if (name == string(second_name))
			return 1;
		if (name == string(third_name))
			return 2;
		throw std::out_of_range("下标错误");
	}

	forward_data& operator=(const forward_data& f)
	{
		this->data_model_base::operator=(f);

		return *this;
	}

	forward_data& operator=(forward_data&& f) noexcept
	{
		this->data_model_base::operator=(std::move(f));

		return *this;
	}

	void generate_time_stamp(float_t exponent_1, float_t exponent_2, float_t interval)
	{
		count = 1 + static_cast<size_t>(std::floor((exponent_2 - exponent_1) / interval));

		data.resize(_data_content_count());
		for (auto& item : data)
			item.resize(count);

		for (size_t i = 0; i < count; ++i)
		{
			const auto exponent = exponent_1 + i * interval;
			(*this)[index_name][i] = static_cast<float_t>(i) + 1;
			(*this)[second_name][i] = pow(10, exponent);
			(*this)[third_name][i] = 0;
		}
	}

	void generate_time_stamp_by_count(float_t exponent_1, float_t exponent_2, size_t count)
	{
		const auto interval = count > 1 ? (exponent_2 - exponent_1) / (count - 1) : 0;
		generate_time_stamp(exponent_1, exponent_2, interval);
	}

	void generate_default_time_stamp()
	{
		generate_time_stamp_by_count(-5, 0, 40);
	}
};

class isometric_model : public data_model_base
{
protected:

	static constexpr char second_name[] = "resistivity";

	virtual void load_layer_height(const json& j)
	{
		if (!j.count(_layer_height()))
		{
			data_model_base::
				throw_critical_data_miss_exception(_layer_height());
		}
		layer_height = j[_layer_height()].get<float_t>();
	}

	std::vector<string> _data_content_name() override { return {index_name, second_name}; }
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
	float_t layer_height;

	isometric_model() : data_model_base()
	{
		layer_height = 0;
	}

	isometric_model(const isometric_model& g) : data_model_base(g)
	{
		layer_height = g.layer_height;
	}

	isometric_model(isometric_model&& g) noexcept : data_model_base(std::move(g))
	{
		layer_height = g.layer_height;
	}

	virtual ~isometric_model() = default;

	size_type get_name_idx(const string& name) const override
	{
		if (name == string(index_name))
			return 0;
		if (name == string(second_name))
			return 1;
		throw std::out_of_range("下标错误");
	}

	float_t get_height() const
	{
		return layer_height;
	}

	isometric_model& operator=(const isometric_model& g)
	{
		this->data_model_base::operator=(g);
		layer_height = g.layer_height;
		return *this;
	}

	isometric_model& operator=(isometric_model&& g) noexcept
	{
		layer_height = g.layer_height;
		this->data_model_base::operator=(std::move(g));
		return *this;
	}
};

class geoelectric_model : public data_model_base
{
	static constexpr char second_name[] = "height";
	static constexpr char third_name[] = "resistivity";

	std::vector<string> _data_content_name() override { return {index_name, second_name, third_name}; }

	void load_additional_data(const json& j) override
	{
	}

	json save_additional_data() override
	{
		json patch_j;
		return patch_j;
	}

public:

	geoelectric_model() = default;

	geoelectric_model(const geoelectric_model& f) = default;

	geoelectric_model(geoelectric_model&& f) noexcept : data_model_base(std::move(f))
	{
	}

	virtual ~geoelectric_model() = default;

	size_type get_name_idx(const string& name) const override
	{
		if (name == string(index_name))
			return 0;
		if (name == string(second_name))
			return 1;
		if (name == string(third_name))
			return 2;
		throw std::out_of_range("下标错误");
	}

	geoelectric_model& operator=(const geoelectric_model& f) = default;

	geoelectric_model& operator=(const isometric_model& i)
	{
		this->data_model_base::operator=(i);

		this->data.clear();
		this->data.resize(this->_data_content_count());
		for (auto& item : this->data)
		{
			item.resize(i.size());
		}

		//convert isometric model to geoelectric model
		data[0] = i.get_item(index_name);
		data[1] = vector(i.size(), i.get_height());
		data[2] = i.get_item(1);

		return *this;
	}

	geoelectric_model& operator=(geoelectric_model&& f) noexcept
	{
		this->data_model_base::operator=(std::move(f));

		return *this;
	}
};

class filter_coefficient
{
public:
	using vector = global::vector;
	using string = std::string;

	vector hkl_coef;
	vector sin_coef;
	vector cos_coef;
	vector gs_coef;

	filter_coefficient() = default;

	filter_coefficient(const filter_coefficient& coef)
	{
		*this = coef;
	}

	~filter_coefficient() = default;

	filter_coefficient& operator=(const filter_coefficient& coef) = default;

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
		catch (std::exception& e)
		{
			std::cerr << e.what() << std::endl;
			return;
		}

		v.clear();
		while (!input_file.eof())
		{
			float_t idx;
			input_file >> idx;
			v.emplace_back(idx);
		}
	}

	vector& get_hkl() { return hkl_coef; }
	vector& get_cos() { return cos_coef; }

	void load_hkl_coef(const string& path)
	{
		load_coef_from_file(path, hkl_coef);
	}

	void load_sin_coef(const string& path)
	{
		load_coef_from_file(path, sin_coef);
	}

	void load_cos_coef(const string& path)
	{
		load_coef_from_file(path, cos_coef);
	}

	void load_gs_coef(const string& path)
	{
		load_coef_from_file(path, gs_coef);
	}

	bool is_valid() const
	{
		return hkl_coef.empty() || sin_coef.empty() || cos_coef.empty() || gs_coef.empty();
	}
};
