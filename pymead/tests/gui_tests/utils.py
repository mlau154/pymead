import time
import typing

import pytest
from PyQt6.QtCore import QTimer, QPointF, QPoint
from PyQt6.QtWidgets import QDialog, QApplication

from pymead.gui.gui import GUI
from pymead.core.point import Point
from pytestqt.qtbot import QtBot


def perform_action_on_dialog(dialog_trigger: typing.Callable,
                             on_dialog_open_action: typing.Callable,
                             time_out: int = 5) -> QDialog:
    """
    Solution for getting active dialog for GUI testing. From
    https://github.com/pytest-dev/pytest-qt/issues/256#issuecomment-1915675942

    Returns the current dialog (active modal widget). If there is no
    dialog, it waits until one is created for a maximum of 5 seconds (by
    default).

    Parameters
    ----------
    dialog_trigger: typing.Callable
        Function that triggers the dialog creation.

    on_dialog_open_action: typing.Callable
        Function that must have a QDialog as its sole argument. This function is called immediately after the
        dialog is found.

    time_out: int
        Maximum time (seconds) to wait for the dialog creation.
    """

    dialog: QDialog = None
    start_time = time.time()

    # Helper function to catch the dialog instance and hide it
    def dialog_creation():
        # Wait for the dialog to be created or timeout
        nonlocal dialog
        while dialog is None and time.time() - start_time < time_out:
            dialog = QApplication.activeModalWidget()

        # Avoid errors when dialog is not created
        if dialog is not None:
            on_dialog_open_action(dialog)

    # Create a thread to get the dialog instance and call dialog_creation trigger
    QTimer.singleShot(1, dialog_creation)
    dialog_trigger()

    # Wait for the dialog to be created or timeout
    while dialog is None and time.time() - start_time < time_out:
        continue

    assert isinstance(
        dialog, QDialog
    ), f"No dialog was created after {time_out} seconds. Dialog type: {type(dialog)}"

    return dialog


@pytest.fixture
def app(qtbot):
    gui = GUI(bypass_vercheck=True)
    gui.show()
    qtbot.addWidget(gui)
    return gui


def pointer(app, point: Point, qtbot: QtBot):
    app.auto_range_geometry()
    x = point.canvas_item.scatter.data[0][0]
    y = point.canvas_item.scatter.data[0][1]
    point_pixel_location = app.airfoil_canvas.getViewBox().mapViewToDevice(QPointF(x, y)).toPoint()
    qtbot.mouseMove(app.airfoil_canvas, point_pixel_location)
    qtbot.wait(100)
    qtbot.mouseMove(app.airfoil_canvas, QPoint(point_pixel_location.x() + 1, point_pixel_location.y() + 1))
    qtbot.wait(100)
    return point
