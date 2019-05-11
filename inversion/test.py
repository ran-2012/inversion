import os
import forward


# def mswig_import_helper():
#     from os.path import dirname
#     import imp
#     fp = None
#     try:
#         fp, pathname, description = imp.find_module('data_py', [dirname(__file__)])
#     except ImportError:
#         import data_py
#         return data_py
#     if fp is not None:
#         try:
#             _mod = imp.load_module('data_py', fp, pathname, description)
#         finally:
#             fp.close()
#         return _mod
#
#
# data_py = swig_import_helper()


def data_test():
    try:
        print('data_test begin')
        d = forward.isometric_model()
        d.load_from_file("../test_data/data_load_test.json")
        print(d.count)
        for i in range(d.count):
            print(d['idx'][i])

        d2 = forward.geoelectric_model()
        d2 = d
        d2.save_to_file('../test_data/data_save_test_geo.json')

    except Exception as e:
        print(e.args)
        print('error')
    finally:
        print("data_test finished")


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
        data = forward.forward_data()

        coef.load_cos_coef('../test_data/cos_xs.txt')
        coef.load_hkl_coef('../test_data/hankel1.txt')
        geo.load_from_file('../test_data/test_geo_model.json')
        data.generate_time_stamp_by_count(-5, 0, 100)

        f.load_general_params(10, 100, 50)
        f.load_filter_coef(coef)
        f.load_geo_model(geo)
        f.load_time_stamp(data)

        f.forward()

        m = f.get_result_late_m()
        e = f.get_result_late_e()

    except Exception as e:
        print(repr(e))
        print('error')


if __name__ == '__main__':
    data_test()
    cuda_test()
    forward_test()
    input("Press Enter to continue...")
