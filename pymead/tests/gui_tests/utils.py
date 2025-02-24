import time
import typing

import pytest
from PyQt6.QtCore import QTimer, QPointF, QPoint
from PyQt6.QtWidgets import QDialog, QApplication
from pynput.mouse import Controller, Button

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
    print(f"Creating GUI...")
    gui = GUI(bypass_vercheck=True, bypass_exit_save_dialog=True)
    print(f"Showing GUI...")
    gui.show()
    print("Adding GUI widget to Qt-Bot...")
    qtbot.addWidget(gui)
    print("Completed GUI generation")
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


def moving_point_2(app, point: Point, qtbot: QtBot):

    mouse = Controller()

    app.auto_range_geometry()

    qtbot.wait(6000)

    x = point.canvas_item.scatter.data[0][0]
    y = point.canvas_item.scatter.data[0][1]

    # Compute the position of the input Point object in global pixel coordinates
    screenGeometry = app.airfoil_canvas.getViewBox().screenGeometry()
    viewRange = app.airfoil_canvas.getViewBox().viewRange()

    print(f"screenGeometry: {screenGeometry}")
    print(f"viewRange: {viewRange}")
    viewNormalizedX = (x - viewRange[0][0]) / (viewRange[0][1] - viewRange[0][0])

    viewNormalizedY = (y - viewRange[1][0]) / (viewRange[1][1] - viewRange[1][0])

    print(f"viewNormalizedX: {viewNormalizedX}")
    print(f"viewNormalizedY: {viewNormalizedY}")

    point_pixel_location = QPointF(
        screenGeometry.topLeft().x() + screenGeometry.width() * viewNormalizedX,
        screenGeometry.topLeft().y() + screenGeometry.height() * (1 - viewNormalizedY) - 25
    ).toPoint()

    print(f"point_pixel_location: {point_pixel_location}")


    # Move pointer relative to current position
    mouse.position = (point_pixel_location.x(), point_pixel_location.y())
    qtbot.wait(300)
    mouse.press(Button.left)
    qtbot.wait(300)
    mouse.move(25, -25)
    mouse.release(Button.left)
    qtbot.wait(400)

    return point
