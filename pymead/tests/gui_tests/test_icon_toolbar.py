import pytest
from PyQt5.QtCore import Qt

from pymead.gui.gui import GUI


@pytest.fixture
def app(qtbot):
    gui = GUI()
    gui.show()
    qtbot.addWidget(gui)
    return gui


def test_background_color_change(app):
    app.main_icon_toolbar.buttons["change-background-color"]["button"].setChecked(False)
    app.main_icon_toolbar.buttons["change-background-color"]["button"].setChecked(True)


def test_grid_button_click(app, qtbot):
    qtbot.mouseClick(app.main_icon_toolbar.buttons["grid"]["button"], Qt.LeftButton)
