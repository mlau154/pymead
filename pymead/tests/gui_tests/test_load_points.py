from pymead.tests.gui_tests.utils import app
from pymead import TEST_DIR

import os


def test_load_points_whitespace(app):

    txt_file = os.path.join(TEST_DIR, "gui_tests", "test_load_points_whitespace.txt")

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        dialog.inputs[1].setValue(txt_file)
        dialog.accept()

    app.load_points(dialog_test_action=dialog_action)

    app.geo_col.clear_container()
