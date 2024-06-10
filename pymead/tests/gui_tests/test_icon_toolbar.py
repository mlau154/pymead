import unittest
import sys

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from pymead.gui.gui import GUI

app = QApplication(sys.argv)
app.processEvents()
app.setStyle('Fusion')


class IconToolbarTest(unittest.TestCase):
    def setUp(self) -> None:
        """Create the GUI"""
        if len(sys.argv) > 1:
            self.form = GUI(path=sys.argv[1])
        else:
            self.form = GUI()

    def test_icons(self):
        self.form.main_icon_toolbar.buttons["change-background-color"]["button"].setChecked(False)
        self.form.main_icon_toolbar.buttons["change-background-color"]["button"].setChecked(True)
        QTest.mouseClick(self.form.main_icon_toolbar.buttons["grid"]["button"], Qt.LeftButton)
