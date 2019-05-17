import numpy as npy
import tensorflow as tf
from tensorflow.python.keras import Model

import forward
import helper

log = helper.get_logger()
log.propagate = False


def forward_result(f: forward.forward_gpu, res: list, height: float):
    iso = forward.isometric_model()
    iso.resize(len(res))
    iso.set_item_s('resistivity', res)
    iso.height = height

    f.load_geo_model(forward.iso_to_geo(iso))
    f.forward()

    m = f.get_result_late_m()

    return m['response']


def inversion():
    f = forward.forward_gpu()

    raw_geo = forward.geoelectric_model()
    raw_geo.load_from_file('../test_data/test_geo_model.json')

    coef = forward.filter_coefficient()
    coef.load_cos_coef('../test_data/cos_xs.txt')
    coef.load_hkl_coef('../test_data/hankel1.txt')

    time = forward.forward_data()
    time.generate_time_stamp_by_count(-5, 0, 100)

    f.load_general_params(10, 100, 50)
    f.load_filter_coef(coef)
    f.load_geo_model(raw_geo)
    f.load_time_stamp(time)

    f.forward()

    m = f.get_result_late_m()
    m.name = 'res late_m'

    helper.add_noise(m, 0.05)
    response_m = m['response']

    res_geo = forward.isometric_model()
    res_geo.resize(5)
    res_geo.height = 50
    res_geo.name = 'inversion'

    height = res_geo.height
    count = len(res_geo)

    def forward_nn(resistivity: npy.ndarray):
        return npy.array(forward_result(f, list(resistivity.tolist()), height))

    class forward_model(tf.keras.Model):
        def __init__(self):
            super(Model, self).__init__()
            # self.resistivity = tf.contrib.eager.Variable(tf.random.uniform([count], minval=50.0, maxval=200))
            self.resistivity = tf.contrib.eager.Variable(tf.fill([count], 150.0))
            # self.resistivity = tf.contrib.eager.Variable([100.0, 50.0, 50.0, 200.0, 200.0])

        def call(self, inputs, **kwargs):
            return forward_nn(self.resistivity)

    def loss_func(inputs, targets):
        r = npy.zeros(npy.shape(inputs))
        for r_i in range(1, len(r)):
            r[r_i] = inputs[r_i] - inputs[r_i - 1]

        r = npy.square(r / height)
        r_loss = npy.sum(r)

        relative_error = (forward_nn(inputs) - npy.array(targets))
        return tf.reduce_mean(tf.square(relative_error / npy.array(targets))) + r_loss

    def grad(forward_model, inputs, targets):
        step = 1.0

        cur_loss = loss_func(forward_model.resistivity.numpy(), targets)

        grads_ = npy.zeros([count])
        for j in range(len(grads_)):
            temp_res = forward_model.resistivity.numpy().copy()
            temp_res[j] += step
            temp_loss = loss_func(temp_res, targets)

            grads_[j] = (temp_loss - cur_loss) / step

        return tf.convert_to_tensor(grads_, dtype=tf.float32), cur_loss

    tf.enable_eager_execution()
    with tf.device('/cpu:0'):
        model = forward_model()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=200)

        for i in range(10):
            grads, loss = grad(model, None, response_m)
            optimizer.apply_gradients(zip([grads], [model.resistivity]),
                                      global_step=tf.train.get_or_create_global_step())

            log.info('loop %d completed, loss = %d' % (i, loss))

        res_geo.set_item_s('resistivity', model.resistivity.numpy().tolist())

    f.load_geo_model(forward.iso_to_geo(res_geo))

    f.forward()
    res_response_m = f.get_result_late_m()

    fig = helper.draw_resistivity(raw_geo, forward.iso_to_geo(res_geo))
    fig.show()

    fig = helper.draw_forward_result(m, res_response_m)
    fig.show()

    input('press any key')


if __name__ == '__main__':
    inversion()

    # def forward_loss(geo, forward_data):
    #     if npy.min(geo) <= 0.:
    #         f_loss = npy.asarray([1.0], dtype=npy.float32)
    #         f_loss = f_loss[0]
    #     else:
    #         forward_data_c = forward_result(f, geo, height)
    #
    #         dc = npy.asarray(forward_data_c)
    #         dr = npy.asarray(forward_data)
    #
    #         f_loss = npy.sum(npy.square(dr - dc) / npy.square(dr))
    #
    #     return f_loss
    #
    # def forward_loss_gradient(geo, forward_data):
    #     step = 1.
    #
    #     cur_loss = forward_loss(geo, forward_data)
    #     grad = npy.ndarray([len(geo)])
    #     for i in range(len(geo)):
    #         geo_grad = npy.array(geo)
    #         geo_grad[i] += step
    #
    #         grad[i] = \
    #             (forward_loss(geo, forward_data) - cur_loss) / step
    #
    # @tf.RegisterGradient("ForwardLossGradient")
    # def forward_loss_gradient_tf(op, grad):
    #     geo = op.inputs[0]
    #     forward_data = op.inputs[1]
    #
    #     grad_g = tf.py_func(forward_loss_gradient,
    #                         inp=[geo, forward_data],
    #                         Tout=tf.float32)
    #     grad_f = tf.zeros(tf.shape(forward_data))
    #
    #     return grad_g * grad, grad_f
    #
    # def forward_loss_tf(geo, forward_data):
    #     with tf.get_default_graph().gradient_override_map(
    #             {"PyFunc": 'ForwardLossGradient'}):
    #         f_loss = tf.py_func(forward_loss,
    #                             inp=[geo, forward_data],
    #                             Tout=tf.float32)
    #     return f_loss
    #
    # tf.enable_eager_execution()
    # response_tf = tf.placeholder(tf.float32, [len(response_m)])
    # init_geo_tf = tf.constant(100, tf.float32, [res_geo.count])
    # geo_tf = tf.Variable(tf.zeros([res_geo.count]))
    # loss = tf.reduce_mean(forward_loss_tf(geo_tf, response_tf))
    #
    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    # trainer = optimizer.minimize(loss)
    #
    # iterate_count = 20
    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #
    #     geo_tf = tf.assign(geo_tf, init_geo_tf)
    #
    #     for it in range(iterate_count):
    #         sess.run(trainer, feed_dict={response_tf: response_m})
    #         log.info('iterate %d over' % it)
    #
    #     res_geo.set_item_s('resistivity', sess.run(geo_tf))
    #     log.info('training over')
