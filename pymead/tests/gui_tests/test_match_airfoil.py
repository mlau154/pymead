from pymead.tests.gui_tests.utils import app


def test_match_airfoil(app):
    airfoil = app.load_example_basic_airfoil_blunt()
    params_container = app.geo_col.container()["params"]
    params_container["Length-6"].v = 0.00126
    params_container["Length-7"].v = 0.00126

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        dialog.inputs[2].setValue('n0012-il')
        dialog.accept()

    app.match_airfoil(dialog_test_action=dialog_action)

