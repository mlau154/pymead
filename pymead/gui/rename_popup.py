from PyQt6.QtWidgets import QDialog, QLineEdit, QVBoxLayout, QDialogButtonBox


class RenamePopup(QDialog):
    def __init__(self, name: str, item_to_rename, parent=None):
        super().__init__(parent)
        self.item_to_rename = item_to_rename
        self.name = name
        self.setWindowTitle("Rename")

        QBtn = QDialogButtonBox.ButtonRole.Ok | QDialogButtonBox.ButtonRole.Cancel

        self.button_box = QDialogButtonBox(QBtn)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout = QVBoxLayout()
        self.q_line_edit = QLineEdit(name)
        self.layout.addWidget(self.q_line_edit)
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)

    def accept(self) -> None:
        self.item_to_rename.setText(0, self.q_line_edit.text())
        self.done(0)

    def reject(self) -> None:
        self.done(0)
