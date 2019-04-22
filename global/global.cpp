
#include "global.h"

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
	void detail::_log(const std::string& tag, const std::string& content)
	{
		std::stringstream msg;
		msg << current_time() << "\t";
		msg << tag << " | " << content << "\n";
#if defined(_MSC_VER)
		OutputDebugString(msg.str().c_str());
#elif
		std::clog << msg.str() << std::flush();
#endif
	}

	std::string current_time()
	{
		std::stringstream time;
		auto now = std::chrono::system_clock::now();
		auto now_c = std::chrono::system_clock::to_time_t(now);
		time << std::put_time(localtime(&now_c), "%Y-%m-%d %H:%M:%S");
		return time.str();
	}
}
