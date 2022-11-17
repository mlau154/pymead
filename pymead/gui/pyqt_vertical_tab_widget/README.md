# pyqt-vertical-tab-widget
PyQt vertical tab widget (text is horizontal)

## Requirements
PyQt5 >= 5.8

## Setup
`python -m pip install pyqt-vertical-tab-widget`

## Example
Code Example
```python
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel
from pyqt_vertical_tab_widget.verticalTabWidget import VerticalTabWidget


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    myWindow = VerticalTabWidget()
    for i in range(3):
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setText(f'Widget {i}')
        label.setFont(QFont('Arial', 30, QFont.Bold))
        myWindow.addTab(label, f'Tab {i}')
    myWindow.show()
    sys.exit(app.exec_())
```

Result

https://user-images.githubusercontent.com/55078043/145132730-b2294734-118d-4fc4-b493-943a1482e607.mp4

