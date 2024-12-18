from pymead.core.point import PointSequence
from pymead.tests.gui_tests.utils import app, pointer
from pymead.utils.misc import convert_rgba_to_hex, get_setting
from pytestqt.qtbot import QtBot

import matplotlib.colors
import numpy as np


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

    point_scatter_color_hex_6_digit_d = convert_rgba_to_hex(bezier.canvas_item.opts["pen"].color().getRgb())[:-2]
    point_scatter_color_setting_d = matplotlib.colors.cnames[get_setting(f"curve_default_pen_color")].lower()
    assert point_scatter_color_hex_6_digit_d == point_scatter_color_setting_d

    app.geo_col.clear_container()


def test_select_deselect_object_ferguson(app):
    le = app.geo_col.add_point(0.0, 0.0)
    upper1 = app.geo_col.add_point(0.0, 0.04)
    upper3 = app.geo_col.add_point(0.6, 0.02)
    upper4 = app.geo_col.add_point(0.9, 0.04)

    ferguson = app.geo_col.add_ferguson(point_sequence=PointSequence(points=[le, upper1, upper3, upper4]))

    app.geo_col.select_object(ferguson)
    assert ferguson.tree_item.isSelected()
    assert not ferguson.tree_item.hoverable

    point_scatter_color_hex_6_digit = convert_rgba_to_hex(ferguson.canvas_item.opts["pen"].color().getRgb())[:-2]
    point_scatter_color_setting = matplotlib.colors.cnames[get_setting(f"curve_selected_pen_color")].lower()
    assert point_scatter_color_hex_6_digit == point_scatter_color_setting

    app.geo_col.deselect_object(ferguson)
    assert not ferguson.tree_item.isSelected()

    point_scatter_color_hex_6_digit_d = convert_rgba_to_hex(ferguson.canvas_item.opts["pen"].color().getRgb())[:-2]
    point_scatter_color_setting_d = matplotlib.colors.cnames[get_setting(f"curve_default_pen_color")].lower()
    assert point_scatter_color_hex_6_digit_d == point_scatter_color_setting_d


def test_select_deselect_object_airfoil_thin(app):
    upper_curve_array = np.array([
        [0.0, 0.0],
        [0.0, 0.05],
        [0.05, 0.05],
        [0.6, 0.04],
        [1.0, 0.0]
    ])
    lower_curve_array = np.array([
        [0.0, -0.05],
        [0.05, -0.05],
        [0.7, 0.01]
    ])

    point_seq_upper = PointSequence([app.geo_col.add_point(xy[0], xy[1]) for xy in upper_curve_array])
    point_seq_lower = PointSequence([point_seq_upper.points()[0],
                                     *[app.geo_col.add_point(xy[0], xy[1]) for xy in lower_curve_array],
                                     point_seq_upper.points()[-1]])

    bez_upper = app.geo_col.add_bezier(point_seq_upper)
    bez_lower = app.geo_col.add_bezier(point_seq_lower)

    airfoil = app.geo_col.add_airfoil(point_seq_upper.points()[0],
                                      point_seq_upper.points()[-1],
                                      upper_surf_end=None,
                                      lower_surf_end=None
                                      )

    app.geo_col.select_object(airfoil)
    assert airfoil.tree_item.isSelected()
    assert not airfoil.tree_item.hoverable

    bez_one_color_hex_6_digit_s = convert_rgba_to_hex(bez_upper.canvas_item.opts["pen"].color().getRgb())[:-2]
    bez_two_color_hex_6_digit_s = convert_rgba_to_hex(bez_lower.canvas_item.opts["pen"].color().getRgb())[:-2]

    bez_color_setting_s = matplotlib.colors.cnames[get_setting(f"curve_selected_pen_color")].lower()
    assert bez_one_color_hex_6_digit_s == bez_color_setting_s
    assert bez_two_color_hex_6_digit_s == bez_color_setting_s

    app.geo_col.deselect_object(airfoil)
    assert not airfoil.tree_item.isSelected()
    assert airfoil.tree_item.hoverable

    bez_one_color_hex_6_digit_d = convert_rgba_to_hex(bez_upper.canvas_item.opts["pen"].color().getRgb())[:-2]
    bez_two_color_hex_6_digit_d = convert_rgba_to_hex(bez_lower.canvas_item.opts["pen"].color().getRgb())[:-2]

    bez_color_setting_d = matplotlib.colors.cnames[get_setting(f"curve_default_pen_color")].lower()
    assert bez_one_color_hex_6_digit_d == bez_color_setting_d
    assert bez_two_color_hex_6_digit_d == bez_color_setting_d

    app.geo_col.clear_container()


def test_select_deselect_object_airfoil_blunt(app):
    point_one_upper = app.geo_col.add_point(0.0, 0.0)
    point_two_upper = app.geo_col.add_point(0.3, 0.2)
    point_three_upper = app.geo_col.add_point(0.6, 0.15)
    point_four_upper = app.geo_col.add_point(0.8, 0.1)

    point_one_lower = app.geo_col.add_point(0.2, -0.1)
    point_two_lower = app.geo_col.add_point(0.5, 0.0)
    point_three_lower = app.geo_col.add_point(0.8, -0.05)

    midpoint = app.geo_col.add_point(0.8, 0.1)

    bezier_one_blunt = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_two_upper,
        point_three_upper,
        point_four_upper
    ]))

    bezier_two_blunt = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_one_lower,
        point_two_lower,
        point_three_lower
    ]))

    line_one_blunt = app.geo_col.add_line(PointSequence(points=[point_four_upper, midpoint]))
    line_two_blunt = app.geo_col.add_line(PointSequence(points=[point_three_lower, midpoint]))

    airfoil_blunt = app.geo_col.add_airfoil(point_one_upper,
                                            midpoint,
                                            point_four_upper,
                                            point_three_lower
                                            )

    app.geo_col.select_object(airfoil_blunt)
    assert airfoil_blunt.tree_item.isSelected()
    assert not airfoil_blunt.tree_item.hoverable

    bez_one_color_hex_6_digit_s = convert_rgba_to_hex(bezier_one_blunt.canvas_item.opts["pen"].color().getRgb())[:-2]
    bez_two_color_hex_6_digit_s = convert_rgba_to_hex(bezier_two_blunt.canvas_item.opts["pen"].color().getRgb())[:-2]
    line_one_color_hex_6_digit_s = convert_rgba_to_hex(line_one_blunt.canvas_item.opts["pen"].color().getRgb())[:-2]
    line_two_color_hex_6_digit_s = convert_rgba_to_hex(line_two_blunt.canvas_item.opts["pen"].color().getRgb())[:-2]

    curve_color_setting_s = matplotlib.colors.cnames[get_setting(f"curve_selected_pen_color")].lower()
    curve_color_setting_d = matplotlib.colors.cnames[get_setting(f"curve_default_pen_color")].lower()
    assert bez_one_color_hex_6_digit_s == curve_color_setting_s
    assert bez_two_color_hex_6_digit_s == curve_color_setting_s
    assert line_one_color_hex_6_digit_s == curve_color_setting_d
    assert line_two_color_hex_6_digit_s == curve_color_setting_d

    app.geo_col.deselect_object(airfoil_blunt)
    assert not airfoil_blunt.tree_item.isSelected()
    assert airfoil_blunt.tree_item.hoverable

    bez_one_color_hex_6_digit_d = convert_rgba_to_hex(bezier_one_blunt.canvas_item.opts["pen"].color().getRgb())[:-2]
    bez_two_color_hex_6_digit_d = convert_rgba_to_hex(bezier_two_blunt.canvas_item.opts["pen"].color().getRgb())[:-2]
    line_one_color_hex_6_digit_d = convert_rgba_to_hex(line_one_blunt.canvas_item.opts["pen"].color().getRgb())[:-2]
    line_two_color_hex_6_digit_d = convert_rgba_to_hex(line_two_blunt.canvas_item.opts["pen"].color().getRgb())[:-2]

    assert bez_one_color_hex_6_digit_d == curve_color_setting_d
    assert bez_two_color_hex_6_digit_d == curve_color_setting_d
    assert line_one_color_hex_6_digit_s == curve_color_setting_d
    assert line_two_color_hex_6_digit_s == curve_color_setting_d

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


def test_clear_selected_objects_lines(app):
    point_one = app.geo_col.add_point(0.2, 0.6)
    point_two = app.geo_col.add_point(0.3, 0.1)
    point_three = app.geo_col.add_point(0.9, 0.1)
    point_four = app.geo_col.add_point(0.4, 0.1)
    line_one = app.geo_col.add_line(point_sequence=PointSequence(points=[point_one, point_two]))
    line_two = app.geo_col.add_line(point_sequence=PointSequence(points=[point_one, point_two]))

    app.geo_col.select_object(line_one)
    app.geo_col.select_object(line_two)

    assert len(app.geo_col.selected_objects["lines"]) == 2
    assert not line_one.tree_item.hoverable
    assert not line_two.tree_item.hoverable

    app.geo_col.clear_selected_objects()

    assert len(app.geo_col.selected_objects["lines"]) == 0
    assert line_one.tree_item.hoverable
    assert line_two.tree_item.hoverable

    app.geo_col.clear_container()


def test_clear_selected_objects_bezier(app):
    point_one = app.geo_col.add_point(0.2, 0.6)
    point_two = app.geo_col.add_point(0.3, 0.1)
    point_three = app.geo_col.add_point(0.9, 0.1)
    point_four = app.geo_col.add_point(0.4, 0.1)
    bez_one = app.geo_col.add_bezier(point_sequence=PointSequence(points=[point_one, point_two, point_three]))
    bez_two = app.geo_col.add_bezier(point_sequence=PointSequence(points=[point_four, point_two, point_three]))

    app.geo_col.select_object(bez_one)
    app.geo_col.select_object(bez_two)

    assert len(app.geo_col.selected_objects["bezier"]) == 2
    assert not bez_one.tree_item.hoverable
    assert not bez_two.tree_item.hoverable

    app.geo_col.clear_selected_objects()

    assert len(app.geo_col.selected_objects["bezier"]) == 0
    assert bez_one.tree_item.hoverable
    assert bez_two.tree_item.hoverable

    app.geo_col.clear_container()


def test_clear_selected_objects_ferguson(app):
    le = app.geo_col.add_point(0.0, 0.0)
    upper1 = app.geo_col.add_point(0.0, 0.04)
    upper3 = app.geo_col.add_point(0.6, 0.02)
    upper4 = app.geo_col.add_point(0.9, 0.04)
    upper5 = app.geo_col.add_point(1, 0.5)

    ferguson_one = app.geo_col.add_ferguson(point_sequence=PointSequence(points=[le, upper1, upper3, upper4]))
    ferguson_two = app.geo_col.add_ferguson(point_sequence=PointSequence(points=[le, upper1, upper3, upper5]))

    app.geo_col.select_object(upper1)
    app.geo_col.select_object(ferguson_two)

    assert len(app.geo_col.selected_objects["ferguson"]) == 1
    assert ferguson_one.tree_item.hoverable
    assert not ferguson_two.tree_item.hoverable

    app.geo_col.clear_selected_objects()

    assert len(app.geo_col.selected_objects["ferguson"]) == 0
    assert ferguson_one.tree_item.hoverable
    assert ferguson_two.tree_item.hoverable


def test_clear_selected_objects_airfoils(app):
    airfoil_container = app.geo_col.container()["airfoils"]

    upper_curve_array = np.array([
        [0.0, 0.0],
        [0.0, 0.05],
        [0.05, 0.05],
        [0.6, 0.04],
        [1.0, 0.0]
    ])
    lower_curve_array = np.array([
        [0.0, -0.05],
        [0.05, -0.05],
        [0.7, 0.01]
    ])

    point_seq_upper = PointSequence([app.geo_col.add_point(xy[0], xy[1]) for xy in upper_curve_array])
    point_seq_lower = PointSequence([point_seq_upper.points()[0],
                                     *[app.geo_col.add_point(xy[0], xy[1]) for xy in lower_curve_array],
                                     point_seq_upper.points()[-1]])

    bez_upper = app.geo_col.add_bezier(point_seq_upper)
    bez_lower = app.geo_col.add_bezier(point_seq_lower)

    airfoil_thin = app.geo_col.add_airfoil(point_seq_upper.points()[0],
                                           point_seq_upper.points()[-1],
                                           upper_surf_end=None,
                                           lower_surf_end=None
                                           )

    point_one_upper = app.geo_col.add_point(0.0, 0.0)
    point_two_upper = app.geo_col.add_point(0.3, 0.2)
    point_three_upper = app.geo_col.add_point(0.6, 0.15)
    point_four_upper = app.geo_col.add_point(0.8, 0.1)

    point_one_lower = app.geo_col.add_point(0.2, -0.1)
    point_two_lower = app.geo_col.add_point(0.5, 0.0)
    point_three_lower = app.geo_col.add_point(0.8, -0.05)

    midpoint = app.geo_col.add_point(0.8, 0.1)

    bezier_one_blunt = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_two_upper,
        point_three_upper,
        point_four_upper
    ]))

    bezier_two_blunt = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_one_lower,
        point_two_lower,
        point_three_lower
    ]))

    line_one_blunt = app.geo_col.add_line(PointSequence(points=[point_four_upper, midpoint]))
    line_two_blunt = app.geo_col.add_line(PointSequence(points=[point_three_lower, midpoint]))

    airfoil_blunt = app.geo_col.add_airfoil(point_one_upper,
                                            midpoint,
                                            point_four_upper,
                                            point_three_lower
                                            )

    app.geo_col.select_object(airfoil_thin)
    app.geo_col.select_object(airfoil_blunt)

    assert len(app.geo_col.selected_objects["airfoils"]) == 2
    assert len(airfoil_container) == 2

    app.geo_col.clear_selected_objects()

    assert len(app.geo_col.selected_objects["airfoils"]) == 0
    assert len(airfoil_container) == 2

    app.geo_col.clear_container()


def test_point_hover(app, qtbot: QtBot):

    point_og = app.geo_col.add_point(0.2, 0.6)
    point = pointer(app, point_og, qtbot)

    point_brush_color_hex_6_digit = convert_rgba_to_hex(point.canvas_item.scatter.opts["brush"].color().getRgb())[:-2]
    point_brush_color_setting = matplotlib.colors.cnames[get_setting(f"scatter_hovered_brush_color")].lower()
    point_pen_color_hex_6_digit = convert_rgba_to_hex(point.canvas_item.scatter.opts["pen"].color().getRgb())[:-2]
    point_pen_color_setting = matplotlib.colors.cnames[get_setting(f"scatter_hovered_pen_color")].lower()
    app.geo_col.clear_container()
    assert point_brush_color_hex_6_digit == point_brush_color_setting
    assert point_pen_color_hex_6_digit == point_pen_color_setting


def test_bezier_hover(app, qtbot: QtBot):
    point_one = app.geo_col.add_point(0.0, 0.0)
    point_two = app.geo_col.add_point(0.5, 0.2)
    point_three = app.geo_col.add_point(1.0, 0.0)
    bezier = app.geo_col.add_bezier(point_sequence=PointSequence(points=[point_one, point_two, point_three]))

    point_og = app.geo_col.add_point(0.5, 0.1)
    point = pointer(app, point_og, qtbot)

    bezier_color_hex_6_digit = convert_rgba_to_hex(bezier.canvas_item.opts["pen"].color().getRgb())[:-2]
    bezier_scatter_color_setting = matplotlib.colors.cnames[get_setting(f"curve_hovered_pen_color")].lower()
    assert bezier_color_hex_6_digit == bezier_scatter_color_setting

    app.geo_col.clear_container()


def test_ferguson_hover(app, qtbot: QtBot):
    le = app.geo_col.add_point(0.0, 0.0)
    upper1 = app.geo_col.add_point(0.0, 0.04)
    upper3 = app.geo_col.add_point(0.6, 0.02)
    upper4 = app.geo_col.add_point(0.9, 0.04)

    ferguson = app.geo_col.add_ferguson(point_sequence=PointSequence(points=[le, upper1, upper3, upper4]))

    point_og = app.geo_col.add_point(0.0, 0.0)
    point = pointer(app, point_og, qtbot)

    bezier_color_hex_6_digit = convert_rgba_to_hex(ferguson.canvas_item.opts["pen"].color().getRgb())[:-2]
    bezier_scatter_color_setting = matplotlib.colors.cnames[get_setting(f"curve_hovered_pen_color")].lower()
    assert bezier_color_hex_6_digit == bezier_scatter_color_setting


def test_airfoil_hover(app, qtbot: QtBot):
    upper_curve_array = np.array([
        [0.0, 0.0],
        [0.0, 0.05],
        [0.05, 0.05],
        [0.6, 0.04],
        [1.0, 0.0]
    ])
    lower_curve_array = np.array([
        [0.0, -0.05],
        [0.05, -0.05],
        [0.7, 0.01]
    ])

    point_seq_upper = PointSequence([app.geo_col.add_point(xy[0], xy[1]) for xy in upper_curve_array])
    point_seq_lower = PointSequence([point_seq_upper.points()[0],
                                     *[app.geo_col.add_point(xy[0], xy[1]) for xy in lower_curve_array],
                                     point_seq_upper.points()[-1]])

    bez_upper = app.geo_col.add_bezier(point_seq_upper)
    bez_lower = app.geo_col.add_bezier(point_seq_lower)

    airfoil_thin = app.geo_col.add_airfoil(point_seq_upper.points()[0],
                                           point_seq_upper.points()[-1],
                                           upper_surf_end=None,
                                           lower_surf_end=None
                                           )

    point_og = app.geo_col.add_point(0.1, 0.0)
    point = pointer(app, point_og, qtbot)

    assert app.airfoil_canvas.airfoil_text_items[airfoil_thin].toPlainText() == "Airfoil-1"

    app.geo_col.clear_container()


def test_remove_pymead_obj_point(app):
    point_container = app.geo_col.container()["points"]
    point_one = app.geo_col.add_point(0.5, 0.4)
    point_two = app.geo_col.add_point(0.7, 0.6)
    app.geo_col.remove_pymead_obj(point_two)

    assert "Point-1" in point_container
    assert "Point-2" not in point_container
    assert len(point_container) == 1

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


def test_remove_pymead_obj_ferguson(app):
    point_container = app.geo_col.container()["points"]
    ferg_container = app.geo_col.container()["ferguson"]

    le = app.geo_col.add_point(0.0, 0.0)
    upper1 = app.geo_col.add_point(0.0, 0.04)
    upper3 = app.geo_col.add_point(0.6, 0.02)
    upper4 = app.geo_col.add_point(0.9, 0.04)
    upper5 = app.geo_col.add_point(1, 0.5)

    ferguson_one = app.geo_col.add_ferguson(point_sequence=PointSequence(points=[le, upper1, upper3, upper4]))
    ferguson_two = app.geo_col.add_ferguson(point_sequence=PointSequence(points=[le, upper1, upper3, upper5]))

    app.geo_col.remove_pymead_obj(ferguson_two)
    app.geo_col.remove_pymead_obj(upper3)
    assert len(point_container) == 4
    assert len(ferg_container) == 0


def test_remove_pymead_obj_airfoil_thin(app):
    airfoil_container = app.geo_col.container()["airfoils"]

    upper_curve_array = np.array([
        [0.0, 0.0],
        [0.0, 0.05],
        [0.05, 0.05],
        [0.6, 0.04],
        [1.0, 0.0]
    ])
    lower_curve_array = np.array([
        [0.0, -0.05],
        [0.05, -0.05],
        [0.7, 0.01]
    ])

    point_seq_upper = PointSequence([app.geo_col.add_point(xy[0], xy[1]) for xy in upper_curve_array])
    point_seq_lower = PointSequence([point_seq_upper.points()[0],
                                     *[app.geo_col.add_point(xy[0], xy[1]) for xy in lower_curve_array],
                                     point_seq_upper.points()[-1]])

    bez_upper = app.geo_col.add_bezier(point_seq_upper)
    bez_lower = app.geo_col.add_bezier(point_seq_lower)

    airfoil = app.geo_col.add_airfoil(point_seq_upper.points()[0],
                                      point_seq_upper.points()[-1],
                                      upper_surf_end=None,
                                      lower_surf_end=None
                                      )
    assert len(airfoil_container) == 1
    app.geo_col.remove_pymead_obj(airfoil)
    assert len(airfoil_container) == 0

    airfoil_two = app.geo_col.add_airfoil(point_seq_upper.points()[0],
                                          point_seq_upper.points()[-1],
                                          upper_surf_end=None,
                                          lower_surf_end=None
                                          )

    assert len(airfoil_container) == 1
    app.geo_col.remove_pymead_obj(bez_lower)
    assert len(airfoil_container) == 0

    bez_lower_two = app.geo_col.add_bezier(point_seq_lower)

    airfoil_three = app.geo_col.add_airfoil(point_seq_upper.points()[0],
                                            point_seq_upper.points()[-1],
                                            upper_surf_end=None,
                                            lower_surf_end=None
                                            )

    assert len(airfoil_container) == 1
    app.geo_col.remove_pymead_obj(point_seq_upper[1])
    app.geo_col.remove_pymead_obj(point_seq_upper[2])
    app.geo_col.remove_pymead_obj(point_seq_upper[0])

    assert len(airfoil_container) == 0

    app.geo_col.clear_container()


def test_remove_pymead_obj_airfoil_blunt(app):
    airfoil_container = app.geo_col.container()["airfoils"]
    bezier_container = app.geo_col.container()["bezier"]
    point_container = app.geo_col.container()["points"]
    line_container = app.geo_col.container()["lines"]

    point_one_upper = app.geo_col.add_point(0.0, 0.0)
    point_two_upper = app.geo_col.add_point(0.3, 0.2)
    point_three_upper = app.geo_col.add_point(0.6, 0.15)
    point_four_upper = app.geo_col.add_point(0.8, 0.1)

    point_one_lower = app.geo_col.add_point(0.2, -0.1)
    point_two_lower = app.geo_col.add_point(0.5, 0.0)
    point_three_lower = app.geo_col.add_point(0.8, -0.05)

    midpoint = app.geo_col.add_point(0.8, 0.1)

    bezier_one_blunt = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_two_upper,
        point_three_upper,
        point_four_upper
    ]))

    bezier_two_blunt = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_one_lower,
        point_two_lower,
        point_three_lower
    ]))

    line_one_blunt = app.geo_col.add_line(PointSequence(points=[point_four_upper, midpoint]))
    line_two_blunt = app.geo_col.add_line(PointSequence(points=[point_three_lower, midpoint]))

    airfoil_blunt = app.geo_col.add_airfoil(point_one_upper,
                                            midpoint,
                                            point_four_upper,
                                            point_three_lower
                                            )

    app.geo_col.remove_pymead_obj(airfoil_blunt)
    assert len(airfoil_container) == 0
    assert len(point_container) == 8
    assert len(bezier_container) == 2
    assert len(line_container) == 2

    airfoil_blunt_two = app.geo_col.add_airfoil(point_one_upper,
                                                midpoint,
                                                point_four_upper,
                                                point_three_lower
                                                )

    app.geo_col.remove_pymead_obj(line_two_blunt)
    assert len(airfoil_container) == 0
    assert len(point_container) == 8
    assert len(bezier_container) == 2
    assert len(line_container) == 1

    line_two_blunt_two = app.geo_col.add_line(PointSequence(points=[point_three_lower, midpoint]))
    airfoil_blunt_three = app.geo_col.add_airfoil(point_one_upper,
                                                  midpoint,
                                                  point_four_upper,
                                                  point_three_lower
                                                  )

    app.geo_col.remove_pymead_obj(midpoint)
    assert len(airfoil_container) == 0
    assert len(point_container) == 7
    assert len(bezier_container) == 2
    assert len(line_container) == 0

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
    point_four = app.geo_col.add_point(0.7, 0.1)
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


def test_remove_selected_objects_airfoils(app):
    airfoil_container = app.geo_col.container()["airfoils"]
    point_container = app.geo_col.container()["points"]
    bezier_container = app.geo_col.container()["bezier"]
    line_container = app.geo_col.container()["lines"]

    upper_curve_array = np.array([
        [0.0, 0.0],
        [0.0, 0.05],
        [0.05, 0.05],
        [0.6, 0.04],
        [1.0, 0.0]
    ])
    lower_curve_array = np.array([
        [0.0, -0.05],
        [0.05, -0.05],
        [0.7, 0.01]
    ])

    point_seq_upper = PointSequence([app.geo_col.add_point(xy[0], xy[1]) for xy in upper_curve_array])
    point_seq_lower = PointSequence([point_seq_upper.points()[0],
                                     *[app.geo_col.add_point(xy[0], xy[1]) for xy in lower_curve_array],
                                     point_seq_upper.points()[-1]])

    bez_upper = app.geo_col.add_bezier(point_seq_upper)
    bez_lower = app.geo_col.add_bezier(point_seq_lower)

    airfoil = app.geo_col.add_airfoil(point_seq_upper.points()[0],
                                      point_seq_upper.points()[-1],
                                      upper_surf_end=None,
                                      lower_surf_end=None
                                      )

    point_one_upper = app.geo_col.add_point(0.0, 0.0)
    point_two_upper = app.geo_col.add_point(0.3, 0.2)
    point_three_upper = app.geo_col.add_point(0.6, 0.15)
    point_four_upper = app.geo_col.add_point(0.8, 0.1)

    point_one_lower = app.geo_col.add_point(0.2, -0.1)
    point_two_lower = app.geo_col.add_point(0.5, 0.0)
    point_three_lower = app.geo_col.add_point(0.8, -0.05)

    midpoint = app.geo_col.add_point(0.8, 0.1)

    bezier_one_blunt = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_two_upper,
        point_three_upper,
        point_four_upper
    ]))

    bezier_two_blunt = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_one_lower,
        point_two_lower,
        point_three_lower
    ]))

    line_one_blunt = app.geo_col.add_line(PointSequence(points=[point_four_upper, midpoint]))
    line_two_blunt = app.geo_col.add_line(PointSequence(points=[point_three_lower, midpoint]))

    airfoil_blunt = app.geo_col.add_airfoil(point_one_upper,
                                            midpoint,
                                            point_four_upper,
                                            point_three_lower
                                            )

    app.geo_col.select_object(bez_upper)
    app.geo_col.select_object(bezier_two_blunt)
    assert len(app.geo_col.selected_objects["bezier"]) == 2
    app.geo_col.remove_selected_objects()
    assert len(app.geo_col.selected_objects["bezier"]) == 0
    assert len(bezier_container) == 2
    assert len(line_container) == 2
    assert len(airfoil_container) == 0
    assert len(point_container) == 16

    bez_upper_two = app.geo_col.add_bezier(point_seq_upper)

    airfoil = app.geo_col.add_airfoil(point_seq_upper.points()[0],
                                      point_seq_upper.points()[-1],
                                      upper_surf_end=None,
                                      lower_surf_end=None
                                      )

    bezier_two_blunt = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_one_lower,
        point_two_lower,
        point_three_lower
    ]))

    airfoil_blunt = app.geo_col.add_airfoil(point_one_upper,
                                            midpoint,
                                            point_four_upper,
                                            point_three_lower
                                            )

    app.geo_col.select_object(point_seq_upper.points()[2])
    app.geo_col.select_object(midpoint)
    app.geo_col.remove_selected_objects()

    assert len(app.geo_col.selected_objects["points"]) == 0
    assert len(bezier_container) == 4
    assert len(line_container) == 0
    assert len(airfoil_container) == 1
    assert len(point_container) == 14

    app.geo_col.clear_container()
