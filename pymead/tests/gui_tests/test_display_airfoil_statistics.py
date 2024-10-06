from pymead.tests.gui_tests.utils import app


def test_display_airfoil_statistics(app):

    airfoil = app.load_example_basic_airfoil_sharp_dv()

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        dialog.accept()

    app.display_airfoil_statistics(dialog_test_action=dialog_action)

