import unittest
import sys

from PyQt5.QtWidgets import QApplication, QDialog

from pymead.tests.gui_tests.utils import perform_action_on_dialog
from pymead.gui.dialogs import WebAirfoilDialog

from pymead.gui.gui import GUI

app = QApplication(sys.argv)
app.processEvents()
app.setStyle('Fusion')


class WebAirfoilGUITest(unittest.TestCase):
    def setUp(self) -> None:
        """Create the GUI"""
        if len(sys.argv) > 1:
            self.gui = GUI(path=sys.argv[1])
        else:
            self.gui = GUI()

    def test_load_n0012(self):

        def dialog_action(dialog: QDialog):
            dialog.accept()

        perform_action_on_dialog(self.gui.main_icon_toolbar.buttons["generate-web-airfoil"]["button"].click,
                                 dialog_action)
        polyline_subcontainer = self.gui.geo_col.container()["polylines"]
        self.assertTrue("n0012-1" in polyline_subcontainer)

    def test_load_sc20612(self):

        def dialog_action(dialog: WebAirfoilDialog):
            dialog.inputs[1].setValue("sc20612-il")
            dialog.accept()

        perform_action_on_dialog(self.gui.main_icon_toolbar.buttons["generate-web-airfoil"]["button"].click,
                                 dialog_action)
        polyline_subcontainer = self.gui.geo_col.container()["polylines"]
        self.assertTrue("sc20612-1" in polyline_subcontainer)
