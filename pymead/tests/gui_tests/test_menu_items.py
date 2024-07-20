from pymead.gui.gui import GUI
from pymead.tests.gui_tests.utils import app
from pymead.utils.read_write_files import load_data


def test_load_examples(app):
    gui = GUI
    menu_json_file = r"../../gui/gui_settings/menu.json"
    menu_dict = load_data(menu_json_file)
    load_example_data = menu_dict["File"]["Load Example"]

    def _test_load_example_recursively(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                _test_load_example_recursively(v)

            else:
                assert hasattr(gui, v)

    _test_load_example_recursively(load_example_data)

