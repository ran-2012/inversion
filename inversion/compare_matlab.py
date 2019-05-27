import traceback
import time

import numpy as npy
import matplotlib.pyplot as plt

import forward
from helper import *


def forward_test():
    try:
        f = forward.forward_gpu()
        coef = forward.filter_coefficient()
        geo = forward.geoelectric_model()
        data = forward.forward_data()
        m_data = forward.forward_data()

        coef.load_cos_coef('../test_data/cos_xs.txt')
        coef.load_hkl_coef('../test_data/hankel1.txt')
        geo.load_from_file('../test_data/forward_model1.json')
        data.generate_time_stamp_by_count(-5, -2, 20)

        f.load_time_stamp(data)
        f.load_geo_model(geo)
        f.load_general_params(10, 100, 50)
        f.load_filter_coef(coef)

        f.forward()

        data = f.get_result_magnetic()
        data.name = 'CUDA'
        m_data.resize(data.count)
        m_data.time = data.time
        m_data.idx = data.idx
        m_data.name = 'Matlab'

        m_data.set_item_s('response',
                          [1.127565, 0.998018, 0.894319, 0.805530, 0.723312, 0.642532, 0.561468, 0.481032, 0.403363,
                           0.330769, 0.265176, 0.207871, 0.159425, 0.119744, 0.088197, 0.063800, 0.045399, 0.031824,
                           0.022005, 0.015023])

        fig = draw_resistivity(geo)
        fig.show()

        fig = draw_forward_result(data, m_data)
        fig.show()

        print('loss:')
        print(loss(m_data, data))

    except Exception as e:
        traceback.print_exc()
        log.error(repr(e))
    finally:
        log.debug('forward_test finished')


if __name__ == '__main__':
    forward_test()

    input("Press Enter to continue...\n")
