#pragma once

#if defined(_MSC_VER) && defined(_WIN64)
#include <atlbase.h>
#include <string>
namespace global
{
	void log(const std::string& tag, const std::string& conetent)
	{
		//输出日志到VS输出窗口
		std::string msg;
		msg = tag + " | " + content + '\n';
		OutputDebugString(msg.c_str());
	}
}
#elif
namespace global
{
	void log(const std::string& tag, const std::string& content)
	{
		// stub
	}
}
#endif

namespace global
{

	//计算过程中使用的浮点类型
	using float_t = double;

	//pi
	constexpr float_t pi = 3.14159265359;
	//mu_0
	constexpr float_t mu0 = 1.25663706e-6;
}
