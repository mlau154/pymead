from pymead.tests.gui_tests.utils import app
from pymead import TEST_DIR

import os


def test_take_screenshot_full_window(app):
    photo_path = os.path.join(TEST_DIR, "gui_tests", "test_image.jpeg")

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        print(dialog.grid_widget)
        selected_window = dialog.grid_widget["window"]["combobox"].setCurrentText("Full Window")
        file_path = dialog.grid_widget["choose_image_file"]["line"].setText(photo_path)

        dialog.accept()

    app.take_screenshot(dialog_test_action=dialog_action)
    app.geo_col.clear_container()


def test_take_screenshot_tree(app):
    pass


def test_take_screenshot_geometry(app):
    pass


def test_take_screenshot_full_console(app):
    pass