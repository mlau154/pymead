from pymead.tests.gui_tests.utils import app
from pymead.utils.misc import convert_rgba_to_hex, get_setting

import matplotlib.colors


def test_select_object(app):
    """
    Tests ``GeometryCollection.select_object``
    """
    point = app.geo_col.add_point(0.5, 0.4)
    app.geo_col.select_object(point)
    assert point.tree_item.isSelected()
    assert not point.tree_item.hoverable
    point_scatter_color_hex_6_digit = convert_rgba_to_hex(point.canvas_item.scatter.opts["brush"].color().getRgb())[:-2]
    point_scatter_color_setting = matplotlib.colors.cnames[get_setting(f"scatter_selected_brush_color")].lower()
    assert point_scatter_color_hex_6_digit == point_scatter_color_setting
    app.geo_col.clear_container()


def test_deselect_object(app):
    point = app.geo_col.add_point(0.5, 0.4)
    app.geo_col.select_object(point)
    assert point.tree_item.isSelected()
    app.geo_col.deselect_object(point)
    assert point.tree_item.hoverable
    point_scatter_color_hex_6_digit = convert_rgba_to_hex(point.canvas_item.scatter.opts["brush"].color().getRgb())[:-2]
    point_scatter_color_setting = matplotlib.colors.cnames[get_setting(f"scatter_default_brush_color")].lower()
    assert point_scatter_color_hex_6_digit == point_scatter_color_setting
    app.geo_col.clear_container()
