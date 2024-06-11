import pytest

from pymead.gui.gui import GUI


@pytest.fixture
def app(qtbot):
    gui = GUI()
    gui.show()
    qtbot.addWidget(gui)
    return gui


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
