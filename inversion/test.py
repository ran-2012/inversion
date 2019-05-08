
import os
import data_py

def test_main():
	try:
		print(u'呃')
		print(data_py.test_func())
		a=data_py.vec_test_func()
		d=data_py.isometric_model()
		d.load_from_file("../data_load_test.json")
		print(d.count)
		for i in range(d.count):
			print(d['idx'][i])
		print(d.version)

		d2=data_py.geoelectric_model()
		d2=d
		d2.save_to_file('../data_save_test_geo.json')

	except Exception as e:
		print('error')
	finally:
		input()

test_main()

