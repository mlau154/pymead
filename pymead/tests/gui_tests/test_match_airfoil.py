from pymead.tests.gui_tests.utils import app
from pytestqt.qtbot import QtBot

import numpy as np


def test_match_airfoil(app, qtbot: QtBot):
    app.load_example_basic_airfoil_blunt_dv()
    params_container = app.geo_col.container()["params"]
    desvar_container = app.geo_col.container()["desvar"]
    params_container["Length-6"].set_value(0.00126)
    params_container["Length-7"].set_value(0.00126)

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        dialog.inputs[2].setValue('n0012-il')
        dialog.accept()

    def info_dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        assert dialog.windowTitle() == "Information"
        dialog.accept()

    app.match_airfoil(dialog_test_action=dialog_action, info_dialog_test_action=info_dialog_action)

    def check_label():
        assert len(app.text_area.toPlainText()) != 0

    qtbot.wait_until(check_label, timeout=90000)

    assert np.isclose(desvar_container["Length-5"].value(), 0.264, atol=1e-3)
