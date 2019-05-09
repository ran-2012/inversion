#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#define LOG(msg) global::log(__FUNCTION__, msg)
#define TIMER(...) global::scoped_timer __t(__FUNCTION__)

namespace global
{
	namespace detail
	{
		void _log(const std::string& tag, const std::string& content) noexcept;
	}

	std::string current_time();

	//将多个变量合并为字符串
	template <typename T>
	std::string msg(const T& t)
	{
		std::stringstream ss;
		ss << t;
		return ss.str();
	}

	//将多个变量合并为字符串
	template <typename T, typename ...Args>
	std::string msg(const T& t, const Args&...args)
	{
		std::stringstream ss;
		ss << t << msg(args...);
		return ss.str();
	}

	//输出错误信息到std::cerr
	template <typename...Args>
	void err(const Args& ...args)
	{
		std::cerr << msg(args...) << std::endl;
	}

	//输出日志到输出窗口，类型T与U必须可序列化
	template <typename T, typename ...Args>
	void log(const T& tag, const Args& ...content) noexcept
	{
		try
		{
			detail::_log(msg(tag), msg(content...));
		}
		catch (std::exception& e)
		{
			detail::_log("log", e.what());
		}
	}

	//作用域计时器，退出作用域时输出时间
	class scoped_timer
	{
	private:
		void* timer;
	public :
		scoped_timer(std::string name);
		~scoped_timer();
	};
}


namespace global
{
	//计算过程中使用的浮点类型
	using float_t = double;

	using vector = std::vector<float_t>;

	//pi
	constexpr float_t pi = 3.14159265359;
	//mu_0
	constexpr float_t mu0 = 4e-7 * pi;
}
