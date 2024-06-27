from pymead.core.geometry_collection import GeometryCollection
from pymead.core.point import PointSequence
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


def test_select_deselect_object_line_segment(app):
    point_one = app.geo_col.add_point(0.5, 0.4)
    point_two = app.geo_col.add_point(0.7, 0.6)
    line_segment = app.geo_col.add_line(point_sequence=PointSequence(points=[point_one, point_two]))

    app.geo_col.select_object(line_segment)
    assert line_segment.tree_item.isSelected()
    assert not line_segment.tree_item.hoverable
    point_scatter_color_hex_6_digit = convert_rgba_to_hex(line_segment.canvas_item.opts["pen"].color().getRgb())[:-2]
    point_scatter_color_setting = matplotlib.colors.cnames[get_setting(f"curve_selected_pen_color")].lower()
    assert point_scatter_color_hex_6_digit == point_scatter_color_setting

    app.geo_col.deselect_object(line_segment)
    assert not line_segment.tree_item.isSelected()
    assert line_segment.tree_item.hoverable
    point_scatter_color_hex_6_digit_d = convert_rgba_to_hex(line_segment.canvas_item.opts["pen"].color().getRgb())[:-2]
    point_scatter_color_setting_d = matplotlib.colors.cnames[get_setting(f"curve_default_pen_color")].lower()
    assert point_scatter_color_hex_6_digit_d == point_scatter_color_setting_d
    app.geo_col.clear_container()


