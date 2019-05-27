import numpy as npy

import random

from helper import log
import forward
import helper


class Inversion:
    init_step = 50
    final_step = 1

    minial_res = 10.0

    def __init__(self, geo_model: forward.isometric_model, f: forward.forward_gpu):
        self.count = geo_model.count
        self.height = geo_model.height
        self.geo_model = npy.array(geo_model.resistivity)

        self.f = f

        self.magnetic = npy.zeros([1])
        self.steps = npy.ones([self.count]) * Inversion.init_step
        self.grads = npy.zeros([self.count])

    def set_initial_geo_model(self, geo_resistivity: npy.ndarray):
        self.geo_model = geo_resistivity

    def set_target_magnetic(self, magnetic):
        self.magnetic = npy.array(magnetic)

    def rand_idx(self):
        return random.randint(2, self.count - 3)

    def forward_func(self):
        return npy.array(forward_result(self.f, list(self.geo_model), self.height))

    def loss_func(self, geo_model: npy.ndarray):
        norm_2_factor = 1000
        norm_inf_factor = 10

        r = npy.zeros(npy.shape(geo_model))

        for r_i in range(1, len(r)):
            r[r_i] = geo_model[r_i] - geo_model[r_i - 1]

        r = npy.square(r / self.height)
        r_loss = npy.sum(r)

        forward_res = self.forward_func()

        relative_error = (forward_res - self.magnetic) / self.magnetic

        loss = 0.0
        loss += r_loss
        loss += norm_2_factor * npy.mean(npy.square(relative_error))
        loss += norm_inf_factor * npy.max(relative_error)
        return loss

    def loss(self):
        return self.loss_func(self.geo_model)

    def rand_grad(self):
        cur_loss = self.loss_func(self.geo_model)

        self.grads = npy.zeros([self.count])

        idx = self.rand_idx()

        step = self.steps[idx]

        temp_geo_model = self.geo_model.copy()
        temp_geo_model[idx] += step
        temp_loss = self.loss_func(temp_geo_model)

        self.grads[idx] = (temp_loss - cur_loss) / step

        return cur_loss

    def update_step(self):

        for k in range(len(self.steps)):
            if self.grads[k] > 0.0:
                self.steps[k] = - self.steps[k] / 2.0

            if npy.abs(self.steps[k]) < Inversion.final_step:
                if self.steps[k] > 0:
                    self.steps[k] = Inversion.final_step
                else:
                    self.steps[k] = -Inversion.final_step

    def update_model(self):

        for k in range(self.count):
            if self.grads[k] < 0.0:
                self.geo_model[k] -= self.steps[k]
            if self.geo_model[k] < Inversion.minial_res:
                self.geo_model[k] = Inversion.minial_res


def forward_result(f: forward.forward_gpu, res: list, height: float):
    iso = forward.isometric_model()
    iso.resize(len(res))
    iso.set_item_s('resistivity', res)
    iso.height = height

    f.load_geo_model(forward.iso_to_geo(iso))
    f.forward()

    m = f.get_result_magnetic()

    return m['response']


def inversion():
    f = forward.forward_gpu()

    real_geo_model = forward.geoelectric_model()
    real_geo_model.load_from_file('../test_data/test_geo_model.json')

    coef = forward.filter_coefficient()
    coef.load_cos_coef('../test_data/cos_xs.txt')
    coef.load_hkl_coef('../test_data/hankel1.txt')

    time = forward.forward_data()
    time.generate_time_stamp_by_count(-5, -2, 20)

    f.load_general_params(10, 100, 50)
    f.load_filter_coef(coef)
    f.load_geo_model(real_geo_model)
    f.load_time_stamp(time)

    f.forward()

    real_response = f.get_result_magnetic()
    real_response.name = real_geo_model.name

    # helper.add_noise(m, 0.05)
    real_response_m = real_response['response']

    inversion_geo_model = forward.isometric_model()
    inversion_geo_model.resize(20)
    inversion_geo_model.height = 10
    inversion_geo_model.name = 'inversion'

    inv = Inversion(inversion_geo_model, f)
    inv.set_initial_geo_model(npy.ones([inversion_geo_model.count]) * 50.0)
    inv.set_target_magnetic(real_response_m)

    for i in range(500):
        log.info('iteration %d ' % i)

        inv.rand_grad()
        inv.update_model()
        inv.update_step()

        log.info('iteration %d, loss = %f' % (i, inv.loss()))
        if i % 10 == 0:
            inversion_geo_model.set_item_s('resistivity', inv.geo_model.tolist())
            inversion_geo_model.name = 'Inversion'
            f.load_geo_model(forward.iso_to_geo(inversion_geo_model))

            f.forward()
            inversion_response = f.get_result_magnetic()
            inversion_response.name = 'Inversion'

            fig = helper.draw_resistivity(real_geo_model, forward.iso_to_geo(inversion_geo_model), last_height=240)
            fig.show()

            fig = helper.draw_forward_result(real_response, inversion_response)
            fig.show()

            print('loss = %f' % helper.loss(real_response, inversion_response))
            pass

        pass
    #
    # def forward_nn(resistivity: npy.ndarray):
    #     return npy.array(forward_result(f, list(resistivity.tolist()), height))
    #
    # class forward_model(tf.keras.Model):
    #     def __init__(self):
    #         super(Model, self).__init__()
    #         # self.resistivity = tf.contrib.eager.Variable(tf.random.uniform([count], minval=50.0, maxval=200))
    #         # self.resistivity = tf.contrib.eager.Variable(tf.fill([count], 50.0))
    #         self.resistivity = tf.contrib.eager.Variable(
    #             [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 200.0, 200.0, 200.0])
    #         # self.resistivity = tf.contrib.eager.Variable(raw_geo.resistivity)
    #
    #     def call(self, inputs, **kwargs):
    #         return forward_nn(self.resistivity)
    #
    # def loss_func(geo_model, magnetic):
    #     norm_2_factor = 1000
    #     norm_inf_factor = 10
    #
    #     r = npy.zeros(npy.shape(geo_model))
    #
    #     for r_i in range(1, len(r)):
    #         r[r_i] = geo_model[r_i] - geo_model[r_i - 1]
    #
    #     r = npy.square(r / height)
    #     r_loss = npy.sum(r)
    #
    #     forward_res = forward_nn(geo_model)
    #
    #     relative_error = (forward_res - npy.array(magnetic)) / npy.array(magnetic)
    #
    #     return norm_2_factor * npy.mean(npy.square(relative_error)) \
    #            + r_loss \
    #            + norm_inf_factor * npy.max(relative_error)
    #
    # def grad(forward_model, inputs, targets):
    #     step = 10.0
    #
    #     cur_loss = loss_func(forward_model.resistivity.numpy(), targets)
    #
    #     grads_ = npy.zeros([count])
    #     for j in range(len(grads_)):
    #         temp_res = forward_model.resistivity.numpy().copy()
    #         temp_res[j] += step
    #         temp_loss = loss_func(temp_res, targets)
    #
    #         grads_[j] = (temp_loss - cur_loss) / step
    #
    #     return tf.convert_to_tensor(grads_, dtype=tf.float32), cur_loss
    #
    # def rand_grad(forward_model, inputs, targets):
    #     step = 10.0
    #
    #     cur_loss = loss_func(forward_model.resistivity.numpy(), targets)
    #
    #     grads_ = npy.zeros([count])
    #
    #     idx = random.randint(2, 6)
    #
    #     temp_res = forward_model.resistivity.numpy().copy()
    #     temp_res[idx] += step
    #     temp_loss = loss_func(temp_res, targets)
    #
    #     grads_[idx] = (temp_loss - cur_loss) / step
    #
    #     return tf.convert_to_tensor(grads_, dtype=tf.float32), cur_loss
    #
    # tf.enable_eager_execution()
    # with tf.device('/cpu:0'):
    #     model = forward_model()
    #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=10)
    #
    #     for i in range(500):
    #         grads, loss = rand_grad(model, None, real_response_m)
    #         optimizer.apply_gradients(zip([grads], [model.resistivity]),
    #                                   global_step=tf.train.get_or_create_global_step())
    #
    #         res_array = model.resistivity.numpy()
    #         for j in range(len(res_array)):
    #             if res_array[j] <= 10.0:
    #                 res_array[j] = 10.0
    #
    #         model.resistivity = tf.contrib.eager.Variable(res_array)
    #
    #         log.info('iteration %d' % i)
    #         log.info('iteration %d completed, loss = %f' % (i, loss))
    #
    #         if i % 50 == 0:
    #             inversion_geo_model.set_item_s('resistivity', model.resistivity.numpy().tolist())
    #             inversion_geo_model.name = 'Inversion'
    #             f.load_geo_model(forward.iso_to_geo(inversion_geo_model))
    #
    #             f.forward()
    #             inversion_response = f.get_result_magnetic()
    #             inversion_response.name = 'Inversion'
    #
    #             fig = helper.draw_resistivity(real_geo_model, forward.iso_to_geo(inversion_geo_model), last_height=240)
    #             fig.show()
    #
    #             fig = helper.draw_forward_result(real_response, inversion_response)
    #             fig.show()
    #
    #             print('loss = %f' % helper.loss(real_response, inversion_response))
    #             pass


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
