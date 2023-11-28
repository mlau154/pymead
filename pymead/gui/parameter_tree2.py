from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QPushButton, QHBoxLayout, QHeaderView
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QModelIndex

from pymead.core.geometry_collection import GeometryCollection
from pymead.core.param2 import Param


class HeaderButtonRow(QHeaderView):
    sigExpandPressed = pyqtSignal()
    sigCollapsePressed = pyqtSignal()

    def __init__(self, parent):
        super().__init__(Qt.Horizontal, parent)
        self.lay = QHBoxLayout(self)
        self.expandButton = QPushButton("Expand All", self)
        self.collapseButton = QPushButton("Collapse All", self)
        self.expandButton.clicked.connect(self.expandButtonPressed)
        self.collapseButton.clicked.connect(self.collapseButtonPressed)
        self.lay.addWidget(self.expandButton)
        self.lay.addWidget(self.collapseButton)
        self.setLayout(self.lay)
        self.setFixedHeight(40)

    @pyqtSlot()
    def expandButtonPressed(self):
        self.sigExpandPressed.emit()

    @pyqtSlot()
    def collapseButtonPressed(self):
        self.sigCollapsePressed.emit()


class ParamButton(QPushButton):
    def __init__(self, param):
        self.param = param
        super().__init__(param.name())
        self.setMaximumWidth(100)
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        print(f"{self.param.name()} clicked!")


class ParameterTree(QTreeWidget):
    def __init__(self, geo_col: GeometryCollection, parent):
        super().__init__(parent)

        self.geo_col = geo_col
        self.geo_col.geo_tree = self

        self.setColumnCount(1)

        self.items = [QTreeWidgetItem(None, [f"{k}"]) for k in self.geo_col.container().keys()]
        self.topLevelDict = {k: i for i, k in enumerate(self.geo_col.container().keys())}
        self.insertTopLevelItems(0, self.items)

        # Make the header
        self.setHeaderLabel("")
        self.headerRow = HeaderButtonRow(self)
        self.headerRow.sigExpandPressed.connect(self.onExpandPressed)
        self.headerRow.sigCollapsePressed.connect(self.onCollapsePressed)
        self.setHeader(self.headerRow)

        # self.header().resizeSection(0, 320)
        self.setMinimumWidth(200)
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)

    def addParam(self, param: Param):
        top_level_item = self.items[self.topLevelDict["params"]]
        child_item = QTreeWidgetItem([""])
        top_level_item.addChild(child_item)
        param.tree_item = child_item
        # index = top_level_item.indexOfChild(child_item)
        self.setItemWidget(child_item, 0, ParamButton(param))

    def removeParam(self, param: Param):
        top_level_item = self.items[self.topLevelDict["params"]]
        top_level_item.removeChild(param.tree_item)
        param.tree_item = None

    def onExpandPressed(self):
        self.expandAll()

    def onCollapsePressed(self):
        self.collapseAll()
