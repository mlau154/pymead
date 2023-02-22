from PyQt5.QtWidgets import QFileDialog, QLineEdit, QPlainTextEdit


def select_directory(parent, line_edit: QLineEdit):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.DirectoryOnly)
    if file_dialog.exec_():
        line_edit.setText(file_dialog.selectedFiles()[0])


def select_json_file(parent, line_edit: QLineEdit):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.AnyFile)
    file_dialog.setNameFilter(parent.tr("JSON Settings Files (*.json)"))
    if file_dialog.exec_():
        line_edit.setText(file_dialog.selectedFiles()[0])


def select_multiple_json_files(self, text_edit: QPlainTextEdit):
    file_dialog = QFileDialog(self)
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_dialog.setNameFilter(self.tr("JSON Settings Files (*.json)"))
    if file_dialog.exec_():
        text_edit.insertPlainText('\n\n'.join(file_dialog.selectedFiles()))


def select_data_file(parent, line_edit: QLineEdit):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.setNameFilter(parent.tr("Data Files (*.txt *.dat *.csv)"))
    if file_dialog.exec_():
        line_edit.setText(file_dialog.selectedFiles()[0])


def select_multiple_data_files(parent, text_edit: QPlainTextEdit):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_dialog.setNameFilter(parent.tr("Data Files (*.txt *.dat *.csv)"))
    if file_dialog.exec_():
        text_edit.insertPlainText('\n\n'.join(file_dialog.selectedFiles()))
