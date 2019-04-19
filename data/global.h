#pragma once

#if defined(_MSC_VER)
#include <atlbase.h>
#elif
#include <iostream>
#endif

#include <ctime>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace global
{
	decltype(auto) current_time()
	{
		auto now = std::chrono::system_clock::now();
		auto now_c = std::chrono::system_clock::to_time_t(now);
		return std::put_time(localtime(&now_c), "%Y-%m-%d %H:%M:%S");
	}

	template<typename T, typename U>
	void log(const T& tag, const U& content)
	{
		//输出日志到VS输出窗口
		std::stringstream msg;
		msg << current_time() << "\t";
		msg << tag << " | " << content << "\n";
#if defined(_MSC_VER)
		OutputDebugString(msg.str().c_str());
#elif
		std::clog << msg.str()<<std::flush();
#endif
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
