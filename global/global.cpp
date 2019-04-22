
#include "global.h"

#if defined(_MSC_VER) && defined(_DEBUG)
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
	namespace detail
	{
		void _log(const std::string& tag, const std::string& content)
		{
			std::stringstream msg;
			msg << current_time() << "\t";
			msg << tag << " | " << content << "\n";
#if defined(_MSC_VER) && defined(_DEBUG)
			OutputDebugString(msg.str().c_str());
#elif
			std::clog << msg.str() << std::flush();
#endif
		}

		class _scoped_timer
		{
			using clock = std::chrono::steady_clock;
			using time_point = std::chrono::time_point<std::chrono::steady_clock>;

			clock clk;
			time_point begin;
			std::string name;

		public:
			_scoped_timer(std::string n) 
			{
				name = n;
				begin.now();
			}

			~_scoped_timer()
			{
				time_point end = clk.now();
				auto t = std::chrono::duration_cast<duration>(end - begin).count/1000;

				std::stringstream msg;
				msg << name << " excuted in " << t << "ms";

				log("timer", msg.str());
			}
		};
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
