from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem
from pyqtgraph import TreeWidget

from pymead.gui.parameter_tree import PymeadTreeWidgetItem
from pymead.tests.gui_tests.utils import app


def test_show_obj_point(app):
    point_container = app.geo_col.container()["points"]
    point = app.geo_col.add_point(0.3, 0.1)

    for k, v in app.parameter_tree.container_titles.items():
        item = QTreeWidgetItem(app.parameter_tree, [v])
        item.setData(0, Qt.ItemDataRole.DisplayRole, k)

    def get_points_container(app):
        for i in range(app.parameter_tree.topLevelItemCount()):
            item_one = app.parameter_tree.topLevelItem(i)
            if item_one.text(0) == app.parameter_tree.container_titles["points"]:
                return item_one

    points_item = get_points_container(app)
    assert points_item.isExpanded()
    assert not points_item.isHidden()
    app.airfoil_canvas.hidePymeadObjs("points")
    app.showHidePymeadObjs("points", False)
    print(points_item.treeWidget())
    print(points_item.isExpanded())

    app.geo_col.clear_container()


def test_show_obj_point_two(app):
    point_container = None
    point = app.geo_col.add_point(0.2, 0.4)
    for i in range(app.parameter_tree.topLevelItemCount()):
        item = app.parameter_tree.topLevelItem(i)
        if item.text(0) == "Points":
            point_container = item
            break

    assert point_container.isExpanded()
    print(point_container.text(0))
    #app.showHidePymeadObjs("points", True)
    #app.airfoil_canvas.hidePymeadObjs('points')
    app.parameter_tree.expandItem(point_container)
    #assert point.canvas_item.hide()
    #assert not point_container.isExpanded()

    app.geo_col.clear_container()


def test_show_hide_obj_lines(app):
    pass


def test_show_hide_obj_bezier(app):
    pass


def test_show_hide_obj_airfoils(app):
    pass


def test_show_hide_all_objs(app):
    pass



