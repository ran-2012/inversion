import traceback

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import forward
from helper import *


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


@timer
def cuda_test():
    g = forward.forward_gpu()
    g.init_cuda_device()
    g.test_cuda_device()
    print("cuda_test finished")


def forward_test():
    try:
        f = forward.forward_gpu()
        coef = forward.filter_coefficient()
        geo = forward.geoelectric_model()
        geo2 = forward.geoelectric_model()
        iso = forward.isometric_model()
        data = forward.forward_data()

        coef.load_cos_coef('../test_data/cos_xs.txt')
        coef.load_hkl_coef('../test_data/hankel1.txt')
        geo.load_from_file('../test_data/test_geo_model.json')
        geo2.load_from_file('../test_data/test_geo_model2.json')
        iso.load_from_file('../test_data/test_iso_model.json')
        data.generate_time_stamp_by_count(-5, 0, 100)

        fig = draw_resistivity(geo, geo2, forward.iso_to_geo(iso), last_height=300)
        fig.show()

        f.load_general_params(10, 100, 50)
        f.load_filter_coef(coef)
        f.load_geo_model(geo)
        f.load_time_stamp(data)

        f.forward()

        m = f.get_result_late_m()
        e = f.get_result_late_e()

        m.name = 'late_m'
        e.name = 'late_e'

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


def tf_test():
    # 生成数据
    rx = np.linspace(-1, 1, 100)
    ry = 2 * rx + np.random.randn(*rx.shape) * 0.3

    # 正向
    x = tf.placeholder("float")
    y = tf.placeholder("float")

    w = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")

    z = tf.multiply(x, w) + b

    # 反向
    cost = tf.reduce_mean(tf.square(y - z))
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # 运行模型
    init = tf.global_variables_initializer()

    training_epochs = 20
    display_step = 2

    with tf.Session() as sess:
        sess.run(init)
        plotdata = {"batch_size": [], "loss": []}

        for epoch in range(training_epochs):
            for (lx, ly) in zip(rx, ry):
                sess.run(optimizer, feed_dict={x: lx, y: ly})

            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={x: rx, y: ry})
                print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(w), "b=", sess.run(b))

                if not (loss == 'NA'):
                    plotdata["batch_size"].append(epoch)
                    plotdata["loss"].append(loss)
        print("Completed")
        print("cost=", sess.run(cost, feed_dict={x: rx, y: ry})), "w=", sess.run(w), "b=", sess.run(b)

        # 画图
        plt.figure(1)
        plt.subplot(211)
        plt.plot(rx, ry, 'ro', label='Raw data')
        plt.plot(rx, sess.run(w) * rx + sess.run(b), label="Fitted line")
        plt.legend()

        plt.subplot(212)
        plt.plot(plotdata["batch_size"], plotdata["loss"], 'b--')
        plt.xlim(0, 19)
        plt.xlabel("Minibatch number")
        plt.ylabel("Loss")
        plt.show()


if __name__ == '__main__':
    data_test()
    cuda_test()
    forward_test()

    input("Press Enter to continue...\n")
