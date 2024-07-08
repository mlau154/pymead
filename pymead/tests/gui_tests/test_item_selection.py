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


def test_select_deselect_object_bezier(app):
    bez_container = app.geo_col.container()["bezier"]
    le = app.geo_col.add_point(0.0, 0.0)
    upper1 = app.geo_col.add_point(0.0, 0.05)
    upper2 = app.geo_col.add_point(0.05, 0.05)
    upper3 = app.geo_col.add_point(0.6, 0.03)
    upper4 = app.geo_col.add_point(0.8, 0.04)

    bezier = app.geo_col.add_bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper4]))

    app.geo_col.select_object(bezier)
    assert bezier.tree_item.isSelected()
    assert not bezier.tree_item.hoverable

    point_scatter_color_hex_6_digit = convert_rgba_to_hex(bezier.canvas_item.opts["pen"].color().getRgb())[:-2]
    point_scatter_color_setting = matplotlib.colors.cnames[get_setting(f"curve_selected_pen_color")].lower()
    assert point_scatter_color_hex_6_digit == point_scatter_color_setting

    app.geo_col.deselect_object(bezier)
    assert not bezier.tree_item.isSelected()
    assert bezier.tree_item.hoverable

    point_scatter_color_hex_6_digit_d = convert_rgba_to_hex(bezier.canvas_item.opts["pen"].color().getRgb())[:-2]
    point_scatter_color_setting_d = matplotlib.colors.cnames[get_setting(f"curve_default_pen_color")].lower()
    assert point_scatter_color_hex_6_digit_d == point_scatter_color_setting_d

    app.geo_col.clear_container()


def test_clear_selected_objects_points(app):
    point_one = app.geo_col.add_point(0.2, 0.6)
    point_two = app.geo_col.add_point(0.3, 0.1)
    point_three = app.geo_col.add_point(0.9, 0.1)

    app.geo_col.select_object(point_one)
    app.geo_col.select_object(point_two)

    assert point_one.tree_item.isSelected()
    assert not point_one.tree_item.hoverable
    assert point_two.tree_item.isSelected()
    assert not point_two.tree_item.hoverable
    assert not point_three.tree_item.isSelected()
    assert point_three.tree_item.hoverable

    point_scatter_color_hex_6_digit_one = convert_rgba_to_hex(
        point_one.canvas_item.scatter.opts["brush"].color().getRgb())[:-2]
    point_scatter_color_hex_6_digit_two = convert_rgba_to_hex(
        point_two.canvas_item.scatter.opts["brush"].color().getRgb())[:-2]
    point_scatter_color_hex_6_digit_three = convert_rgba_to_hex(
        point_three.canvas_item.scatter.opts["brush"].color().getRgb())[:-2]
    point_scatter_color_setting_select = matplotlib.colors.cnames[get_setting(f"scatter_selected_brush_color")].lower()
    point_scatter_color_setting_default = matplotlib.colors.cnames[get_setting(f"scatter_default_brush_color")].lower()

    assert point_scatter_color_hex_6_digit_one == point_scatter_color_setting_select
    assert point_scatter_color_hex_6_digit_two == point_scatter_color_setting_select
    assert point_scatter_color_hex_6_digit_three == point_scatter_color_setting_default

    app.geo_col.clear_selected_objects()

    assert not point_one.tree_item.isSelected()
    assert point_one.tree_item.hoverable
    assert not point_two.tree_item.isSelected()
    assert point_two.tree_item.hoverable
    assert not point_three.tree_item.isSelected()
    assert point_three.tree_item.hoverable

    point_scatter_color_hex_6_digit_one = convert_rgba_to_hex(
        point_one.canvas_item.scatter.opts["brush"].color().getRgb())[:-2]
    point_scatter_color_hex_6_digit_two = convert_rgba_to_hex(
        point_two.canvas_item.scatter.opts["brush"].color().getRgb())[:-2]
    point_scatter_color_hex_6_digit_three = convert_rgba_to_hex(
        point_three.canvas_item.scatter.opts["brush"].color().getRgb())[:-2]

    assert point_scatter_color_hex_6_digit_one == point_scatter_color_setting_default
    assert point_scatter_color_hex_6_digit_two == point_scatter_color_setting_default
    assert point_scatter_color_hex_6_digit_three == point_scatter_color_setting_default

    app.geo_col.clear_container()


def test_remove_pymead_obj_lines(app):
    line_container = app.geo_col.container()["lines"]
    point_container = app.geo_col.container()["points"]

    point_one = app.geo_col.add_point(0.5, 0.4)
    point_two = app.geo_col.add_point(0.7, 0.6)
    point_three = app.geo_col.add_point(0.4, 0.1)
    line_segment = app.geo_col.add_line(point_sequence=PointSequence(points=[point_one, point_two]))
    line_segment_two = app.geo_col.add_line(point_sequence=PointSequence(points=[point_one, point_three]))

    app.geo_col.remove_pymead_obj(line_segment)
    assert "Line-1" not in line_container
    assert len(line_container) == 1
    assert len(point_container) == 3

    app.geo_col.remove_pymead_obj(point_three)

    assert len(line_container) == 0
    assert len(point_container) == 2
    assert len(app.geo_col.selected_objects["points"]) == 0
    assert len(app.geo_col.selected_objects["lines"]) == 0

    app.geo_col.clear_container()


def test_remove_pymead_obj_bezier(app):
    point_container = app.geo_col.container()["points"]
    bez_container = app.geo_col.container()["bezier"]
    le = app.geo_col.add_point(0.0, 0.0)
    upper1 = app.geo_col.add_point(0.0, 0.05)
    upper2 = app.geo_col.add_point(0.05, 0.05)
    upper3 = app.geo_col.add_point(0.6, 0.03)
    upper4 = app.geo_col.add_point(0.8, 0.04)
    bezier = app.geo_col.add_bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper4]))

    app.geo_col.remove_pymead_obj(upper1)
    app.geo_col.remove_pymead_obj(upper2)
    assert len(point_container) == 3
    assert len(bez_container) == 1

    app.geo_col.remove_pymead_obj(upper3)
    assert len(bez_container) == 0

    app.geo_col.clear_container()


def test_remove_selected_objects_points(app):
    point_container = app.geo_col.container()["points"]
    point_one = app.geo_col.add_point(0.5, 0.4)
    point_two = app.geo_col.add_point(0.7, 0.6)
    point_three = app.geo_col.add_point(0.4, 0.1)

    app.geo_col.select_object(point_one)
    app.geo_col.select_object(point_two)
    assert len(app.geo_col.selected_objects["points"]) == 2
    app.geo_col.remove_selected_objects()

    assert len(app.geo_col.selected_objects["points"]) == 0
    assert len(point_container) == 1
    assert "Point-1" not in point_container
    assert "Point-2" not in point_container
    assert "Point-3" in point_container

    app.geo_col.clear_container()


def test_remove_selected_objects_lines(app):
    line_container = app.geo_col.container()["lines"]
    point_container = app.geo_col.container()["points"]
    point_one = app.geo_col.add_point(0.5, 0.4)
    point_two = app.geo_col.add_point(0.7, 0.6)
    point_three = app.geo_col.add_point(0.4, 0.1)
    point_four = app.geo_col.add_point(0.7,0.1)
    line_segment = app.geo_col.add_line(point_sequence=PointSequence(points=[point_one, point_two]))
    line_segment_two = app.geo_col.add_line(point_sequence=PointSequence(points=[point_three, point_four]))

    app.geo_col.select_object(line_segment)
    assert len(app.geo_col.selected_objects["points"]) == 0
    assert len(app.geo_col.selected_objects["lines"]) == 1
    app.geo_col.remove_selected_objects()

    assert len(line_container) == 1
    assert len(point_container) == 4
    assert "Line-1" not in line_container
    assert "Line-2" in line_container

    app.geo_col.select_object(point_four)
    app.geo_col.select_object(point_one)
    assert len(app.geo_col.selected_objects["points"]) == 2
    assert len(app.geo_col.selected_objects["lines"]) == 0
    app.geo_col.remove_selected_objects()

    assert len(line_container) == 0
    assert len(point_container) == 2

    app.geo_col.clear_container()


def test_remove_selected_objects_bezier(app):

    point_container = app.geo_col.container()["points"]
    bez_container = app.geo_col.container()["bezier"]

    le = app.geo_col.add_point(0.0, 0.0)
    upper1 = app.geo_col.add_point(0.0, 0.05)
    upper2 = app.geo_col.add_point(0.05, 0.05)
    upper3 = app.geo_col.add_point(0.6, 0.03)
    upper4 = app.geo_col.add_point(0.8, 0.04)
    bezier = app.geo_col.add_bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper4]))

    point_one = app.geo_col.add_point(0.1, 0.1)
    point_two = app.geo_col.add_point(0.2, 0.3)
    point_three = app.geo_col.add_point(0.51, 0.7)
    bezier_two = app.geo_col.add_bezier(point_sequence=PointSequence(points=[point_two, point_one, point_three]))

    app.geo_col.select_object(upper1)
    app.geo_col.select_object(upper3)
    app.geo_col.select_object(upper4)

    assert len(app.geo_col.selected_objects["points"]) == 3
    assert len(app.geo_col.selected_objects["bezier"]) == 0

    app.geo_col.remove_selected_objects()

    assert len(point_container) == 5
    assert len(bez_container) == 1
    assert len(app.geo_col.selected_objects["points"]) == 0
    assert len(app.geo_col.selected_objects["bezier"]) == 0
    assert "Bezier-1" not in bez_container

    app.geo_col.select_object(bezier_two)
    app.geo_col.remove_selected_objects()

    assert len(point_container) == 5
    assert len(bez_container) == 0

    app.geo_col.clear_container()
