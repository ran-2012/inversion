import traceback

import matplotlib.pyplot as plt

import forward
from helper import *


def forward_test():
    try:
        f = forward.forward_gpu()
        coef = forward.filter_coefficient()
        geo = forward.geoelectric_model()
        data = forward.forward_data()

        geo_path = ['../test_data/forward_model1.json',
                    '../test_data/forward_model2.json',
                    '../test_data/forward_model3.json']

        geos = []
        responses = []
        for i in range(len(geo_path)):
            geos.append(forward.geoelectric_model())
            responses.append(forward.forward_data())

        coef.load_cos_coef('../test_data/cos_xs.txt')
        coef.load_hkl_coef('../test_data/hankel1.txt')
        data.generate_time_stamp_by_count(-5, 0, 80)

        f.load_general_params(10, 100, 50)
        f.load_filter_coef(coef)
        f.load_time_stamp(data)

        for i in range(len(geo_path)):
            geos[i].load_from_file(geo_path[i])

            f.load_geo_model(geos[i])
            f.forward()

            responses[i] = f.get_result_late_m()
            responses[i].name = geos[i].name

        fig = draw_resistivity(*geos)
        plt.savefig(figure=fig, fname='geo.png')

        fig = draw_forward_result(*responses)
        plt.savefig(figure=fig, fname='response.png')
        # fig.show()

        # fig = draw_forward_result(m)
        #
        # fig.show()

    except Exception as e:
        traceback.print_exc()
        log.error(repr(e))
    finally:
        log.debug('forward_test finished')


if __name__ == '__main__':
    forward_test()

    input("Press Enter to continue...\n")
