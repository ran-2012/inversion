#pragma once

#include <string>
#include <sstream>

namespace global
{
	namespace detail
	{
		void _log(const std::string& tag, const std::string& content);
	}
	std::string current_time();

	//输出日志到输出窗口，类型T与U必须可序列化
	template<typename T, typename U>
	void log(const T& tag, const U& content)
	{
		std::stringstream tag_s, content_s;
		tag_s << tag;
		content_s << content;
		detail::_log(tag_s.str(), content_s.str());
	}
}


namespace global
{

	//计算过程中使用的浮点类型
	using float_t = double;

	//pi
	constexpr float_t pi = 3.14159265359;
	//mu_0
	constexpr float_t mu0 = 1.25663706e-6;
}
