from pymead.tests.gui_tests.utils import app
from pymead import TEST_DIR

import os


def test_take_screenshot(app):
    photo_path = os.path.join(TEST_DIR, "gui_tests", "test_image.jpeg")

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        combobox = dialog.grid_widget["window"]["combobox"]

        for index in range(combobox.count()):
            window = combobox.itemText(index)
            combobox.setCurrentIndex(index)
        file_path = dialog.grid_widget["choose_image_file"]["line"].setText(photo_path)
        dialog.accept()

    def info_dialog(dialog):
        dialog.setModal(False)
        dialog.show()
        assert dialog.windowTitle() == "Information"
        dialog.accept()

    app.take_screenshot(dialog_test_action=dialog_action, info_dialog_action=info_dialog)
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
        print()
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
            #find way to check if the end has changed to .jpg
            dialog.show()
            assert dialog.windowTitle() == "Information"
            dialog.accept()

    app.take_screenshot(dialog_test_action=dialog_action, info_dialog_action=info_dialog)
    app.geo_col.clear_container()
