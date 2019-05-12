import matplotlib.pyplot as plt
import time
import logging as log
from forward import *


def set_default_log():
    log.basicConfig(format='%(asctime)s   %(levelname)s   %(funcName)s   %(message)s',
                    datefmt='%H:%M:%S',
                    level=log.DEBUG)


def process_resistivity(height: list, resistivity: list, final_height: float):
    height_idx = -1
    height_total = 0

    height_ret = []
    res_ret = []

    for res_idx in range(len(resistivity)):
        res_ret.append(resistivity[res_idx])
        res_ret.append(resistivity[res_idx])

        if height_idx == -1:
            height_ret.append(0)
        else:
            height_ret.append(height_total)

        height_idx = height_idx + 1
        height_total += height[height_idx]

        if height_idx == len(resistivity):
            height_ret.append(final_height)
        else:
            height_ret.append(height_total)
    return height_ret, res_ret


def draw_resistivity(*model_list, **kwargs):

    last_height = 200
    if 'last_height' in kwargs:
        last_height = kwargs['last_height']

    draw_list = []
    for g in model_list:
        draw_item = process_resistivity(g['height'], g['resistivity'], last_height)
        draw_list.append(draw_item)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, xlabel='height', ylabel='height')

    for draw_item, g in zip(draw_list, model_list):
        ax.plot(draw_item[0], draw_item[1], label=g.name)

    ax.legend()
    return fig


def draw_forward_result(forward_result: forward_data, *args):
    result_list = [forward_result]
    if len(args) > 0:
        result_list += args

    plt.xlabel('time')
    plt.ylabel('response')

    for f in result_list:
        plt.loglog(forward_result['time'], forward_result['response'], label=f.name)

    plt.legend()
    plt.show()


def timer(f):
    def _timer():
        s = time.clock()
        res = f()
        e = time.clock()
        log.info('%s executed in %f s' % (f.__name__, e - s))
        return res

    return _timer
