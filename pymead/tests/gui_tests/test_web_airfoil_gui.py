from pymead.tests.gui_tests.utils import app


def test_load_n0012(app):
    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        dialog.accept()

    app.airfoil_canvas.generateWebAirfoil(dialog_test_action=dialog_action)
    polyline_subcontainer = app.geo_col.container()["polylines"]
    assert "n0012-1" in polyline_subcontainer
    app.geo_col.clear_container()


def test_load_sc20612(app):
    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        dialog.inputs[1].setValue("sc20612-il")
        dialog.accept()

    app.airfoil_canvas.generateWebAirfoil(dialog_test_action=dialog_action)
    polyline_subcontainer = app.geo_col.container()["polylines"]
    assert "sc20612-1" in polyline_subcontainer
    app.geo_col.clear_container()


def test_load_garbage(app):
    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        dialog.inputs[1].setValue("a;lhkne5lk naekefr")
        dialog.accept()

    def error_dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        assert dialog.windowTitle() == "Error"
        dialog.accept()

    app.airfoil_canvas.generateWebAirfoil(dialog_test_action=dialog_action, error_dialog_action=error_dialog_action)
