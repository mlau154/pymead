import time
import typing

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog, QApplication

from pymead.gui.gui import GUI


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
    gui = GUI()
    gui.show()
    qtbot.addWidget(gui)
    return gui
