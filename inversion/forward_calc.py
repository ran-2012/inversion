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
        data.name = 'Model'

        fig = draw_resistivity(geo)
        fig.show()

        fig = draw_forward_result(data)
        fig.show()


    except Exception as e:
        traceback.print_exc()
        log.error(repr(e))
    finally:
        log.debug('forward_test finished')


if __name__ == '__main__':
    forward_test()

