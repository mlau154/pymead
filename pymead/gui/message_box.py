import typing

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

from pymead.gui.dialogs import PymeadMessageBox


def disp_message_box(message: str, parent: QWidget, theme: dict, message_mode: str = 'error', rich_text: bool = False,
                     dialog_test_action: typing.Callable = None):
    """
    Displays a custom message box

    Parameters
    ----------
    message: str
        Message to display
    parent: QWidget
        Parent widget
    theme: dict
        Current GUI theme
    message_mode: str
        Type of message to send (either 'error', 'info', or 'warn')
    rich_text: bool
        Whether to display the message using rich text
    dialog_test_action: typing.Callable or None
        If not ``None``, a function pointer should be specified that intercepts the call to ``exec`` and
        performs custom actions. Used for testing only.
    """
    window_title_keys = {"error": "Error", "info": "Information", "warn": "Warning"}
    msg_box = PymeadMessageBox(parent, msg=message, window_title=window_title_keys[message_mode], msg_mode=message_mode,
                               theme=theme)
    if rich_text:
        msg_box.setTextFormat(Qt.TextFormat.RichText)
    msg_box.setText(message)
    if dialog_test_action is not None and not dialog_test_action(msg_box):
        return
    msg_box.exec()
