
import data_py

def test_main():
	d=data_py.geoelectric_model()
	d.load_from_file("../data_load_test.json")
	print(d.data);
	print(d.version)

test_main()