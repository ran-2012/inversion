#pragma once

#include <vector>

#include "../global/global.h"

namespace gpu
{
	using float_t = global::float_t;
	using vector = global::vector;

	void init_cuda_device();

	void test_cuda_device();

	/**
	 * \brief 计算正演kernel函数
	 * \param a 回线半径(m)
	 * \param i0 发射电流(A)
	 * \param h 发射、接收回线高度(m)
	 * \param cosine 余弦变换系数
	 * \param hankel 汉克尔变换系数
	 * \param resistivity 地层电阻率(Om)
	 * \param height 地层厚度(m)
	 * \param time 时间(s)
	 * \param magnetic 磁场强度(nT)
	 * \param a_resistivity_late_m 晚期磁场视电阻率
	 * \param a_resistivity_late_e 晚期感应电动势视电阻率
	 */
	void forward(float_t a, float_t i0, float_t h,
	             const vector& cosine, const vector& hankel,
	             const vector& resistivity, const vector& height,
	             const vector& time,
	             vector& magnetic,
	             vector& a_resistivity_late_m,
	             vector& a_resistivity_late_e);
}
