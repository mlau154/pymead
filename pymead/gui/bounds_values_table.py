from PyQt6.QtCore import Qt, QSize
from pyqtgraph import TableWidget

from pymead.core.geometry_collection import GeometryCollection


class BoundsValuesTable(TableWidget):
    def __init__(self, geo_col: GeometryCollection):
        super().__init__(editable=True, sortable=False)
        data = [[dv.name(), f"{dv.lower():.10f}", f"{dv.value():.10f}", f"{dv.upper():.10f}"]
                for dv in geo_col.container()["desvar"].values()]
        self.geo_col = geo_col
        self.initializing_data = True
        self.setData(data)
        self.setHorizontalHeaderLabels(["DesVar Name", "Lower Bound", "Value", "Upper Bound"])
        self.setMinimumHeight(350)
        for column in (0, 2):
            self.makeColumnReadOnly(column)
        self.initializing_data = False

    def handleItemChanged(self, item):
        super().handleItemChanged(item)

        if not self.initializing_data:
            # Get the design variable from the geometry collection
            dv = self.geo_col.container()["desvar"][self.item(item.row(), 0).data(0)]

            # Check if the string input is castable to a float. If not, set the float_val to None
            if item.data(0).strip().replace(".", "").replace("-", "").isnumeric():
                float_val = float(item.data(0).strip())
            else:
                float_val = None

            # Lower bound
            if item.column() == 1:
                # Only change the design variable's lower bound if the input is castable to a float and is less than
                # or equal to the design variable's value
                if float_val is not None and float_val <= dv.value():
                    dv.set_lower(float_val)
                item.setData(0, f"{dv.lower():.10f}")

            # Upper bound
            elif item.column() == 3:
                # Only change the design variable's upper bound if the input is castable to a float and is greater than
                # or equal to the design variable's value
                if float_val is not None and float_val >= dv.value():
                    dv.set_upper(float_val)
                item.setData(0, f"{dv.upper():.10f}")

    def makeColumnReadOnly(self, column: int):
        for row in range(self.rowCount()):
            item = self.item(row, column)
            item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)

    def sizeHint(self):
        width = sum([self.columnWidth(i) for i in range(self.columnCount())])
        width += self.verticalHeader().sizeHint().width()
        width += self.verticalScrollBar().sizeHint().width()
        width += self.frameWidth() * 2
        return QSize(width + 12, self.height())

    def data(self):
        rows = list(range(self.rowCount()))
        columns = list(range(self.columnCount()))
        data = []
        if self.horizontalHeadersSet:
            row = []
            if self.verticalHeadersSet:
                row.append("")

            for c in columns:
                row.append(self.horizontalHeaderItem(c).text())
            data.append(row)

        for r in rows:
            row = []
            if self.verticalHeadersSet:
                row.append(self.verticalHeaderItem(r).text())
            for c in columns:
                item = self.item(r, c)
                if item is not None and item.value != "":
                    row.append(item.value)
                else:
                    row.append(None)
            data.append(row)

        return data
