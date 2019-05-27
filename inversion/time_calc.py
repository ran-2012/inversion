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
        data.generate_time_stamp_by_count(-5, 0, 100)

        f.load_time_stamp(data)
        f.load_geo_model(geo)
        f.load_general_params(10, 100, 50)
        f.load_filter_coef(coef)
        
        f.forward()

        counts = [40, 60, 80, 100]
        layers = [5, 6, 8, 10, 12]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, xlabel='layer count',
                             ylabel='time/s')
        for c in counts:
            data.generate_time_stamp_by_count(-5, 0, c)
            f.load_time_stamp(data)

            t = []

            for l in layers:
                geo.resize(l)
                res = npy.ones([l]) * 100
                hei = npy.ones([l]) * 50
                geo.set_item_s('resistivity', list(res))
                geo.set_item_s('height', list(hei))

                f.load_geo_model(geo)
                s = time.clock()
                f.forward()
                e = time.clock()
                t.append(e - s)

            ax.plot(layers, t, label=str(c), marker='+')

        ax.legend()
        plt.savefig(figure=fig, fname='compare')

    except Exception as e:
        traceback.print_exc()
        log.error(repr(e))
    finally:
        log.debug('forward_test finished')


if __name__ == '__main__':
    forward_test()

    input("Press Enter to continue...\n")
