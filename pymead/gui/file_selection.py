from PyQt5.QtWidgets import QFileDialog, QLineEdit, QPlainTextEdit


def single_file_output_rule(dlg: QFileDialog, line_edit: QLineEdit = None):
    if line_edit is not None:
        if dlg.exec_():
            line_edit.setText(dlg.selectedFiles()[0])
    else:
        return dlg


def multi_file_output_rule(dlg: QFileDialog, text_edit: QPlainTextEdit = None):
    if text_edit is not None:
        if dlg.exec_():
            text_edit.insertPlainText('\n\n'.join(dlg.selectedFiles()))
    else:
        return dlg


def select_directory(parent, line_edit: QLineEdit = None, starting_dir: str = None):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.DirectoryOnly)
    if starting_dir is not None:
        file_dialog.setDirectory(starting_dir)
    return single_file_output_rule(file_dialog, line_edit)


def select_json_file(parent, line_edit: QLineEdit = None):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.AnyFile)
    file_dialog.setNameFilter(parent.tr("JSON Settings Files (*.json)"))
    return single_file_output_rule(file_dialog, line_edit)


def select_jpg_file(parent, line_edit: QLineEdit = None):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.AnyFile)
    file_dialog.setNameFilter(parent.tr("JPEG Files (*.jpg *.jpeg)"))
    return single_file_output_rule(file_dialog, line_edit)


def select_existing_json_file(parent, line_edit: QLineEdit = None):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.setNameFilter(parent.tr("JSON Settings Files (*.json)"))
    return single_file_output_rule(file_dialog, line_edit)


def select_multiple_json_files(parent, text_edit: QPlainTextEdit = None):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_dialog.setNameFilter(parent.tr("JSON Settings Files (*.json)"))
    return multi_file_output_rule(file_dialog, text_edit)


def select_existing_jmea_file(parent, line_edit: QLineEdit = None):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.setNameFilter(parent.tr("JMEA Parametrization (*.jmea)"))
    return single_file_output_rule(file_dialog, line_edit)


def select_data_file(parent, line_edit: QLineEdit = None):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.setNameFilter(parent.tr("Data Files (*.txt *.dat *.csv)"))
    return single_file_output_rule(file_dialog, line_edit)


def select_multiple_data_files(parent, text_edit: QPlainTextEdit = None):
    file_dialog = QFileDialog(parent)
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_dialog.setNameFilter(parent.tr("Data Files (*.txt *.dat *.csv)"))
    return multi_file_output_rule(file_dialog, text_edit)
