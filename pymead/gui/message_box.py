from PyQt5.QtWidgets import QMessageBox


def disp_message_box(message: str, parent, message_mode: str = 'error'):
    msg_box = QMessageBox(parent)
    msg_box.setFont(parent.font())
    msg_box.setText(message)
    if message_mode == 'error':
        msg_box.setWindowTitle('Error')
        msg_box.setIcon(QMessageBox.Critical)
    msg_box.exec()
