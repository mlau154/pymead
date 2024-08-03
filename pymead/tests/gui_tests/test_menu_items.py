from pymead.tests.gui_tests.utils import app
from pymead.utils.read_write_files import load_data
from pymead import GUI_SETTINGS_DIR
import os


def test_load_examples(app):
    menu_json_file = os.path.join(GUI_SETTINGS_DIR, "menu.json")
    menu_dict = load_data(menu_json_file)
    load_example_data = menu_dict["File"]["Load Example"]

    def _test_load_example_recursively(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                _test_load_example_recursively(v)

            else:
                print(v)
                #assert getattr(app, v)()

    _test_load_example_recursively(load_example_data)

