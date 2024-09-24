from pymead.tests.gui_tests.utils import app
from pymead.utils.read_write_files import load_data


def test_edit_bounds_lower(app):
    app.load_example_basic_airfoil_sharp_dv()

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        og = dialog.bv_table.item(0, 1).text()
        dialog.bv_table.item(0, 1).setText(str(0.001))
        dialog.accept()
        assert app.geo_col.container()["desvar"][dialog.bv_table.item(0, 0).text()].lower() == float(
            dialog.bv_table.item(0,1).text())
        dialog.bv_table.item(0, 1).setText(og)

    app.edit_bounds(dialog_test_action=dialog_action)


def test_edit_bounds_bad_name(app):
    app.load_example_basic_airfoil_sharp_dv()

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        desvar_lower_first = app.geo_col.container()["desvar"][dialog.bv_table.item(0, 0).text()].lower()
        dialog.bv_table.item(0, 1).setText("adwehqwe1")
        desvar_lower_second = app.geo_col.container()["desvar"][dialog.bv_table.item(0, 0).text()].lower()
        dialog.accept()
        assert desvar_lower_first == desvar_lower_second

    app.edit_bounds(dialog_test_action=dialog_action)


def test_upper_bound(app):
    app.load_example_basic_airfoil_sharp_dv()

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        og = dialog.bv_table.item(0, 3).text()
        dialog.bv_table.item(0, 3).setText(str(0.5))
        dialog.accept()
        assert app.geo_col.container()["desvar"][dialog.bv_table.item(0, 0).text()].upper() == float(
            dialog.bv_table.item(0, 3).text())
        dialog.bv_table.item(0, 3).setText(og)

    app.edit_bounds(dialog_test_action=dialog_action)


def test_lower_bound_too_high(app):
    app.load_example_basic_airfoil_sharp_dv()

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        og = dialog.bv_table.item(0, 1).text()
        dialog.bv_table.item(0, 1).setText(str(0.5))
        dialog.accept()
        assert (dialog.bv_table.item(0, 1).text() == og)

    app.edit_bounds(dialog_test_action=dialog_action)


def test_upper_bound_too_low(app):
    app.load_example_basic_airfoil_sharp_dv()

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        og = dialog.bv_table.item(0, 3).text()
        dialog.bv_table.item(0, 3).setText(str(0.002))
        dialog.accept()
        assert (dialog.bv_table.item(0, 3).text() == og)

    app.edit_bounds(dialog_test_action=dialog_action)
