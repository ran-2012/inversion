import traceback

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import forward
from helper import *

# 数据加载测试
def data_test():
    try:
        print('data_test begin')
        d = forward.isometric_model()
        d.load_from_file("../test_data/data_load_test.json")
        print(d.count)
        for i in range(d.count):
            print(d['idx'][i])

        d2 = forward.iso_to_geo(d)
        d2.save_to_file('../test_data/data_save_test_geo.json')

    except Exception as e:
        traceback.print_exc()
        log.error(repr(e))
    finally:
        log.debug("data_test finished")

# cuda测试
@timer
def cuda_test():
    g = forward.forward_gpu()
    g.init_cuda_device()
    g.test_cuda_device()
    print("cuda_test finished")

# 正演测试
def forward_test():
    try:
        # 正演类
        f = forward.forward_gpu()
        coef = forward.filter_coefficient()
        geo = forward.geoelectric_model()
        geo2 = forward.geoelectric_model()
        iso = forward.isometric_model()
        data = forward.forward_data()

        # 各种数据加载
        coef.load_cos_coef('../test_data/cos_xs.txt')
        coef.load_hkl_coef('../test_data/hankel1.txt')
        geo.load_from_file('../test_data/test_geo_model.json')
        geo2.load_from_file('../test_data/test_geo_model2.json')
        iso.load_from_file('../test_data/test_iso_model.json')
        data.generate_time_stamp_by_count(-5, -2, 20)

        # 测试绘制地电模型
        fig = draw_resistivity(geo, geo2, forward.iso_to_geo(iso), last_height=300)
        fig.show()

        # 正演类加载数据
        f.load_general_params(10, 100, 50)
        f.load_filter_coef(coef)
        f.load_geo_model(geo)
        f.load_time_stamp(data)

        # 正演开始
        f.forward()

        # 获得结果
        m = f.get_result_late_m()
        e = f.get_result_late_e()

        m.name = 'late_m'
        e.name = 'late_e'

        # 绘制结果
        fig = draw_forward_result(m, e)
        fig.show()

        add_noise(m, 0.1)
        fig = draw_forward_result(m)

        fig.show()

    except Exception as e:
        traceback.print_exc()
        log.error(repr(e))
    finally:
        log.debug('forward_test finished')


if __name__ == '__main__':
    data_test()
    cuda_test()
    forward_test()

    input("Press Enter to continue...\n")
