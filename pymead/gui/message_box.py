from PyQt6.QtCore import Qt

from pymead.gui.dialogs import PymeadMessageBox


def disp_message_box(message: str, parent, theme: dict, message_mode: str = 'error', rich_text: bool = False):
    window_title_keys = {"error": "Error", "info": "Information", "warn": "Warning"}
    msg_box = PymeadMessageBox(parent, msg=message, window_title=window_title_keys[message_mode], msg_mode=message_mode,
                               theme=theme)
    if rich_text:
        msg_box.setTextFormat(Qt.TextFormat.RichText)
    msg_box.setText(message)
    msg_box.exec()
