
import data_py
import os

def test_main():
	print(u'呃')
	print(data_py.test_func())
	d=data_py.geoelectric_model()
	d.load_from_file("../data_load_test.json")
	print(d.count)
	for i in range(d.count):
		print(d[i])
	print(d.version)

	print(d[0])
	d[0]=(0,0)
	print(d[0])
	print(d.count)
	

test_main()