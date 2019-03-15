#pragma once

#include "requirements.h"
#include <iostream>

template<typename T>
class data_model_base
{
	using json=nlohmann::json;

	using string=std::string;
	using pair=std::pair<T,T>;
	using vector=std::vector<pair>;

	using size_t=unsigned int;

	static constexpr char default_version[] = "0.1.0";
	static constexpr char default_comment[] = "";

	virtual string  _version() { return string("version"); }
	virtual string _name() { return string("name"); }
	virtual string _comment() { return string("comment"); }
	virtual string _count() { return string("count"); }
	virtual string _data() { return string("data"); }
	virtual string _first_name() { return string(""); };
	virtual string _second_name() { return string(""); };

	virtual void load_additional_data(const json& j) = 0;
	virtual json save_additional_data() = 0;

public:
	string version;
	string name;
	string comment;

	size_t count;
	vector data;

	data_model_base()
	{
		count = 0;
	}
	virtual ~data_model_base(){}
	
	virtual void load_from_file(const string& path) final
	{
		std::ifstream input_file;
		json j;
		try
		{
			input_file.open(path);
		}
		catch (std::exception &e)
		{
			std::cerr << e.what() << std::endl;
			std::cerr << "文件打开失败" << std::endl;
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
			std::cerr << "加载JSON失败" << " " << path << std::endl;
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
			output_file.open(path);
		}
		catch (std::exception &e)
		{
			std::cerr << e.what() << std::endl;
			std::cerr << "文件打开失败" << std::endl;
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
			std::cerr << "文件写入失败" << " " << path << std::endl;
		}
	}
	virtual void load_from_json(const json& j)
	{
		try
		{
			version = j[_version()].get<string>();
		}
		catch (json::type_error &e)
		{
			std::cerr << e.what() << std::endl;
			version = default_version;
		}
		name = j[_name()].get<string>();
		try
		{
			comment = j[_comment()].get<string>();
		}
		catch (json::type_error &e)
		{
			std::cerr << e.what() << std::endl;
			comment = default_comment;
		}
		count = j[_count()].get<T>();

		data.clear();
		json data_j = j[_data()];
		for(auto it=data_j.begin();it!=data_j.end();++it)
		{
			T first_data = (*it)[_first_name()].get<T>();
			T second_data = (*it)[_second_name()].get<T>();
			data.emplace_back(first_data, second_data);
		}
		if(count!=data.size())
		{
			std::cerr << "Json中count与实际不符" << std::endl;
			count = data.size();
		}
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
		for(auto it=data.begin();it!=data.end();++it)
		{
			json unit_j;
			unit_j[_first_name()] = it->first;
			unit_j[_second_name()] = it->second;
			data_j.emplace_back(unit_j);
		}
		j[_data()] = std::move(data_j);

		json patch_j = save_additional_data();
		return j.patch(patch_j);
	}
};
