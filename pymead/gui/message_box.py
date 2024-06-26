import typing

from PyQt6.QtCore import Qt

from pymead.gui.dialogs import PymeadMessageBox


def disp_message_box(message: str, parent, theme: dict, message_mode: str = 'error', rich_text: bool = False,
                     dialog_test_action: typing.Callable = None):
    window_title_keys = {"error": "Error", "info": "Information", "warn": "Warning"}
    msg_box = PymeadMessageBox(parent, msg=message, window_title=window_title_keys[message_mode], msg_mode=message_mode,
                               theme=theme)
    if rich_text:
        msg_box.setTextFormat(Qt.TextFormat.RichText)
    msg_box.setText(message)
    if dialog_test_action is not None and not dialog_test_action(msg_box):
        return
    msg_box.exec()
