import time
import logging
import matplotlib.pyplot as plt
import numpy as npy

from forward import forward_data


def get_logger():
    _log = logging.getLogger('local')
    _log.setLevel(logging.DEBUG)
    _log.propagate = False

    formatter = logging.Formatter(fmt='%(asctime)s   %(levelname)s   %(funcName)s   %(message)s',
                                  datefmt='%H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    _log.addHandler(handler)

    return _log


log = get_logger()


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

        if height_idx == len(resistivity) - 1:
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
    ax = fig.add_subplot(1, 1, 1, xlabel='height/m', ylabel='resistivity/Ωm',
                         ylim=[0, 300])

    for draw_item, g in zip(draw_list, model_list):
        ax.plot(draw_item[0], draw_item[1], label=g.name)

    ax.legend()
    return fig


def draw_forward_result(forward_result: forward_data, *args):
    result_list = [forward_result]
    if len(args) > 0:
        result_list += args

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, xlabel='time/s', ylabel='B/nT')

    for f in result_list:
        ax.loglog(f['time'], f['response'], label=f.name)

    ax.legend()
    return fig


def timer(f):
    def _timer():
        s = time.clock()
        res = f()
        e = time.clock()
        log.info('%s executed in %f s' % (f.__name__, e - s))
        return res

    return _timer


def add_noise(f: forward_data, ratio=0.05):
    response = f['response']
    for i in range(len(response)):
        response[i] += npy.random.normal(0, response[i] * ratio, )

    f.set_item_s('response', response)


def loss(f_1: forward_data, f_2: forward_data):
    n_1 = npy.array(f_1['response'])
    n_2 = npy.array(f_2['response'])

    return npy.mean(npy.square((n_1 - n_2) / n_1))
