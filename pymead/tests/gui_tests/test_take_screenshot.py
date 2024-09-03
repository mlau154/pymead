from functools import partial

from pymead.tests.gui_tests.utils import app
from pymead import TEST_DIR

import os


def test_take_screenshot(app):
    def dialog_action(dialog, _index: int):
        dialog.setModal(False)
        dialog.show()
        combobox = dialog.grid_widget["window"]["combobox"]

        photo_path = os.path.join(TEST_DIR, "gui_tests", f"test_image_{combobox.itemText(_index)}.jpeg")
        window = combobox.itemText(_index)
        combobox.setCurrentIndex(_index)
        file_path = dialog.grid_widget["choose_image_file"]["line"].setText(photo_path)
        dialog.accept()

    def info_dialog(dialog):
        dialog.setModal(False)
        dialog.show()
        assert dialog.windowTitle() == "Information"
        dialog.accept()

    index = 0
    while True:
        try:
            app.take_screenshot(dialog_test_action=partial(dialog_action, _index=index), info_dialog_action=info_dialog)
            index += 1

        except KeyError:
            break

    app.geo_col.clear_container()
    #check that all of them are actually checked


def test_take_screenshot_bad_path(app):
    photo_path = os.path.join(TEST_DIR, "gui_sts", "test_image.pdh!")

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        selected_window = dialog.grid_widget["window"]["combobox"].setCurrentText("Full Window")
        file_path = dialog.grid_widget["choose_image_file"]["line"].setText(photo_path)
        dialog.accept()

    def error_dialog(dialog):
        dialog.setModal(False)
        dialog.show()
        dialog.accept()

    app.take_screenshot(dialog_test_action=dialog_action, error_dialog_action=error_dialog)
    app.geo_col.clear_container()


def test_take_screenshot_bad_extension(app):
    photo_path = os.path.join(TEST_DIR, "gui_tests", "test_image.pdh!")

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        selected_window = dialog.grid_widget["window"]["combobox"].setCurrentText("Full Window")
        file_path = dialog.grid_widget["choose_image_file"]["line"].setText(photo_path)

    def info_dialog(dialog):
        dialog.setModal(False)
        dialog.show()
        assert dialog.windowTitle() == "Information"
        dialog.accept()

    file_name = app.take_screenshot(dialog_test_action=dialog_action, info_dialog_action=info_dialog)
    assert os.path.splitext(file_name)[1] == '.jpg'
    app.geo_col.clear_container()
