from pymead import EXAMPLES_DIR
import os

from pymead.core.geometry_collection import GeometryCollection
from copy import deepcopy
from pymead.utils.read_write_files import load_data


def test_load_constraints_forward():
    for file in os.listdir(EXAMPLES_DIR):
        extension = os.path.splitext(file)[-1]
        full_path = os.path.join(EXAMPLES_DIR, file)
        if extension == ".jmea":
            geo_col = GeometryCollection.set_from_dict_rep(load_data(full_path))
            print(full_path)
            geocons = deepcopy(list(geo_col.container()["geocon"].keys()))
            for geocon in geocons:
                if geocon not in geo_col.container()["geocon"]:
                    continue
                geo_col.remove_pymead_obj(geo_col.container()["geocon"][geocon])
                print(geocon)
                geo_col.verify_all()


def test_load_constraints_backward():
    for file in os.listdir(EXAMPLES_DIR):
        extension = os.path.splitext(file)[-1]
        full_path = os.path.join(EXAMPLES_DIR, file)
        if extension == ".jmea":
            geo_col = GeometryCollection.set_from_dict_rep(load_data(full_path))
            geocons = deepcopy(list(geo_col.container()["geocon"].keys())[::-1])
            for geocon in geocons:
                if geocon not in geo_col.container()["geocon"]:
                    continue
                geo_col.remove_pymead_obj(geo_col.container()["geocon"][geocon])
            geo_col.verify_all()