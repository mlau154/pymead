# pyqt-vertical-tab-widget
PyQt vertical tab widget (text is horizontal)

## Requirements
PyQt6

## Setup
`python -m pip install pyqt-vertical-tab-widget`

## Example
Code Example
```python
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QLabel
from pyqt_vertical_tab_widget.verticalTabWidget import VerticalTabWidget


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    myWindow = VerticalTabWidget()
    for i in range(3):
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setText(f'Widget {i}')
        label.setFont(QFont('Arial', 30, QFont.Weight.Bold))
        myWindow.addTab(label, f'Tab {i}')
    myWindow.show()
    sys.exit(app.exec())
```

Result

https://user-images.githubusercontent.com/55078043/145132730-b2294734-118d-4fc4-b493-943a1482e607.mp4

