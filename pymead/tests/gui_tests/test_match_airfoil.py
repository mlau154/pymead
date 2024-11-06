from pymead.tests.gui_tests.utils import app

import shutil
import time
import warnings


def test_match_airfoil(app):
    app.load_example_basic_airfoil_blunt_dv()
    params_container = app.geo_col.container()["params"]
    params_container["Length-6"].v = 0.00126
    params_container["Length-7"].v = 0.00126

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        dialog.inputs[2].setValue('n0012-il')
        dialog.accept()

    app.match_airfoil(dialog_test_action=dialog_action)

    t_increment, max_time = 0.1, 60.0
    t = 0.0

    start_time = time.time()
    while (time.time() - start_time) < max_time:
        if len(app.text_area.toPlainText()) > 1:
            text_box = app.text_area.toPlainText()
            print(text_box)
            break
        else:
            time.sleep(t_increment)

    if len(app.text_area.toPlainText()) == 0:
        print("Test timed out")

    #how do hit enter second time after test runs (get rid of info box)
    #not correctly entering if block (not correctly getting text from console?)
