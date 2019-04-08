#pragma once

#include <type_traits>
#include <typeinfo>
#include <ios>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <exception>
#include <utility>

#include <nlohmann/json.hpp>

#include "global.h"

//地电模型与正演结果基类
template<typename FloatT = global::float_t>
class data_model_base
{
protected:
	using string=std::string;
	using vector=std::vector<FloatT>;
	using size_type=typename vector::size_type;
	using json=nlohmann::json;

	static constexpr char default_version[] = "0.1.0";
	static constexpr char default_comment[] = "";

	//检查模板参数是否为浮点类型
	void check_type()
	{
		if (!std::is_floating_point<FloatT>::value)
		{
			std::stringstream msg;
			msg << "模板参数FloatT不是浮点数，";
			msg << "其中 FloatT = ";
			msg << typeid(FloatT).name();
			throw std::invalid_argument(msg.str());
		}
	}
	static void throw_critical_data_miss_exception(const string& data_name)
	{
		std::stringstream msg;
		msg << "关键数据";
		msg << data_name;
		msg << "不存在";
		throw std::runtime_error(msg.str());
	}
	static void ordinary_data_miss(const string& data_name)
	{
		std::stringstream msg;
		msg << "数据";
		msg << data_name;
		msg << "不存在";
		std::cerr << msg.str() << std::endl;
	}

	virtual void load_version(const json& j)
	{
		if(!j.count(_version()))
		{
			ordinary_data_miss(_version());
			version = default_version;
			return;
		}
		version = j[_version()].get<string>();
	}
	virtual void load_name(const json& j)
	{
		if(!j.count(_name()))
		{
			throw_critical_data_miss_exception(_name());
		}
		name = j[_name()].get<string>();
	}
	virtual void load_comment(const json& j)
	{
		if(!j.count(_comment()))
		{
			ordinary_data_miss(_comment());
			comment = default_comment;
			return;
		}
		comment = j[_comment()].get<string>();
	}
	virtual void load_count(const json& j)
	{
		if(!j.count(_count()))
		{
			throw_critical_data_miss_exception(_count());
		}
		count = j[_count()].get<size_type>();
	}
	virtual void load_data(const json& j)
	{
		if(!j.count(_data()))
		{
			throw_critical_data_miss_exception(_data());
		}
		json data_j = j[_data()];
		auto data_names = _data_content_name();
		data.clear();
		data.resize(_data_content_count());
		for (auto& item : data_j)
		{
			for(size_type i=0; i < _data_content_count(); ++i)
			{
				auto temp = item[data_names[i]].get<FloatT>();
				data[i].push_back(temp);
			}
		}
		if (count != data[0].size())
		{
			std::cerr << "JSON中的count与实际数据不符" << std::endl;
			count = data[0].size();
		}
	}

	virtual string _version() { return string("version"); }
	virtual string _name() { return string("name"); }
	virtual string _comment() { return string("comment"); }
	virtual string _count() { return string("count"); }
	virtual string _data() { return string("data"); }

	virtual size_type _data_content_count() { return _data_content_name().size(); }
	virtual std::vector<string> _data_content_name() { return { "" }; }

	virtual void load_additional_data(const json& j) {}
	virtual json save_additional_data() { return json(); }

public:
	string version;
	string name;
	string comment;

	size_type count;
	std::vector<vector> data;

	data_model_base()
	{
		check_type();
		version = default_version;
		comment = default_comment;
		count = 0;
	}
	data_model_base(const data_model_base<FloatT>& d)
	{
		check_type();
		*this = d;
	}
	data_model_base(data_model_base<FloatT>&& d) noexcept
	{
		*this = std::move(d);
	}
	virtual ~data_model_base() = default;

	string get_content_name(size_type idx)
	{
		return _data_content_name()[idx];
	}
	data_model_base<FloatT>& operator=(const data_model_base<FloatT>& d)
	{
		version = d.version;
		name = d.name;
		comment = d.comment;
		count = d.count;
		data = d.data;
		return *this;
	}
	data_model_base<FloatT>& operator=(data_model_base<FloatT>&& d) noexcept
	{
		version = std::move(d.version);
		name = std::move(d.name);
		comment = std::move(d.comment);
		count = d.count;
		data = std::move(d.data);
		return *this;
	}
	virtual vector& operator[](size_type id)
	{
		return data[id];
	}
	virtual void set_item(size_type idx, const vector& p)
	{
		if (idx >= data.size())
		{
			throw(std::runtime_error("idx超过data的数据范围"))
		}
		data[idx] = p;
	}

	virtual size_type size()
	{
		return data.size();
	}
	virtual void load_from_file(const string& path) final
	{
		std::ifstream input_file;
		json j;
		try
		{
			input_file.open(path, std::ifstream::in);
			if(!input_file)
			{
				std::stringstream msg;
				msg << "文件";
				msg << path;
				msg << "不存在";
				throw std::runtime_error(msg.str());
			}
		}
		catch (std::exception & e)
		{
			std::cerr << e.what() << std::endl;
			std::cerr << "无法打开文件 " << path << std::endl;
			return;
		}
		try
		{
			input_file >> j;
			input_file.close();
		}
		catch (std::exception &e)
		{
			std::cerr << e.what() << std::endl;
			std::cerr << "无法加载JSON " << path << std::endl;
			return;
		}
		load_from_json(j);
	}
	virtual void save_to_file(const string& path) final
	{
		std::ofstream output_file;
		auto j = save_to_json();
		try
		{
			output_file.open(path, std::ofstream::out);
		}
		catch (std::exception &e)
		{
			std::cerr << e.what() << std::endl;
			std::cerr << "无法打开文件" << path << std::endl;
			return;
		}
		try
		{
			output_file << j.dump(4);
			output_file.close();
		}
		catch (std::exception &e)
		{
			std::cerr << e.what() << std::endl;
			std::cerr << "无法写入文件 " << path << std::endl; 
		}
	}
	virtual void load_from_json(const json& j)
	{
		load_version(j);
		load_name(j);
		load_comment(j);
		load_count(j);
		load_data(j);

		load_additional_data(j);
	}
	virtual void save_to_json(json& j)
	{
		j = save_to_json();
	}
	virtual json save_to_json()
	{
		json j;
		j[_version()] = version;
		j[_name()] = name;
		j[_comment()] = comment;
		j[_count()] = count;

		json data_j;
		auto names = _data_content_name();
		for(size_type idx=0; idx < data[0].size(); ++idx)
		{
			json unit_j;
			for(size_type i=0; i<_data_content_count(); ++i)
			{
				unit_j[names[i]] = data[i][idx];
			}
			data_j.emplace_back(unit_j);
		}
		j[_data()] = std::move(data_j);

		json patch_j = save_additional_data();
		j.merge_patch(patch_j);
		return j;
	}
};
