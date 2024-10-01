from pymead import EXAMPLES_DIR
import os

from pymead.tests.gui_tests.utils import app
from pymead.utils.read_write_files import load_data


def test_load_constraints(app):
    for file in os.listdir(EXAMPLES_DIR):
        extension = os.path.splitext(file)[-1]
        full_path = os.path.join(EXAMPLES_DIR, file)
        if extension in [".jmea", ".json", ".pkl"]:
            current_file = app.geo_col.set_from_dict_rep(load_data(full_path))
            current_file.verify_all()

