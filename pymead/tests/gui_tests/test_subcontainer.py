from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem
from pyqtgraph import TreeWidget

from pymead.gui.parameter_tree import PymeadTreeWidgetItem
from pymead.tests.gui_tests.utils import app


def test_show_obj(app):
    point_container = app.geo_col.container()["points"]
    point = app.geo_col.add_point(0.3, 0.1)

    for k, v in app.parameter_tree.container_titles.items():
        item = QTreeWidgetItem(app.parameter_tree, [v])
        item.setData(0, Qt.ItemDataRole.DisplayRole, k)

    def get_points_container(app):
        for i in range(app.parameter_tree.topLevelItemCount()):
            item = app.parameter_tree.topLevelItem(i)
            if item.text(0) == app.parameter_tree.container_titles["points"]:
                return item

    points_item = get_points_container(app)
    assert points_item.isExpanded()
    app.airfoil_canvas.hidePymeadObjs("points")
    #points_item = get_points_container(app)
    #assert not points_item.isExpanded()
    app.geo_col.clear_container()
