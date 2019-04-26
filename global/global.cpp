
#include "global.h"

#if defined(_MSC_VER) && defined(_DEBUG)
#include <atlbase.h>
#else
#include <iostream>
#endif

#include <ctime>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <iostream>

namespace global
{
	namespace detail
	{
		void _log(const std::string& tag, const std::string& content) noexcept
		{
			try
			{
				std::stringstream msg;
				msg << current_time() << "\t";
				msg << tag << " | " << content << "\n";
#if defined(_MSC_VER) && defined(_DEBUG)
				OutputDebugString(msg.str().c_str());
#else
				std::clog << msg.str() << std::flush;
#endif

			}
			catch (std::exception& e)
			{
				std::cerr << e.what() << std::endl;
			}
		}

		class _scoped_timer
		{
			using clock = std::chrono::steady_clock;
			using time_point = std::chrono::time_point<std::chrono::steady_clock>;
			using duration=std::chrono::milliseconds;

			clock clk;
			time_point begin;
			std::string name;

		public:
			_scoped_timer(std::string n = std::string()) 
			{
				name = n;
				begin = clk.now();
			}

			~_scoped_timer()
			{
				time_point end = clk.now();
				auto t = std::chrono::duration_cast<duration>(end - begin).count()/1000;

				std::stringstream msg;
				msg << name << " executed in " << t << "ms";

				log("timer", msg.str());
			}
		};
	}

	scoped_timer::scoped_timer(std::string name)
	{
		timer = static_cast<void*>(new detail::_scoped_timer(name));
	}

	scoped_timer::~scoped_timer()
	{
		delete static_cast<detail::_scoped_timer*>(timer);
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
