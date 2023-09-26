from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt


def disp_message_box(message: str, parent, message_mode: str = 'error', rich_text: bool = False):
    msg_box = QMessageBox(parent)
    msg_box.setFont(parent.font())
    if rich_text:
        msg_box.setTextFormat(Qt.RichText)
    msg_box.setText(message)
    if message_mode == 'error':
        msg_box.setWindowTitle('Error')
        msg_box.setIcon(QMessageBox.Critical)
    elif message_mode == 'info':
        msg_box.setWindowTitle('Information')
        msg_box.setIcon(QMessageBox.Information)
    elif message_mode == 'warn':
        msg_box.setWindowTitle('Warning')
        msg_box.setIcon(QMessageBox.Warning)
    msg_box.exec()
