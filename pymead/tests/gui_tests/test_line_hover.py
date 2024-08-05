import matplotlib
from pytestqt.qtbot import QtBot

from pymead.core.point import PointSequence
from pymead.tests.gui_tests.utils import pointer, app
from pymead.utils.misc import convert_rgba_to_hex, get_setting


def test_line_hover(app, qtbot: QtBot):
    point_one = app.geo_col.add_point(0.0, 0.0)
    point_two = app.geo_col.add_point(0.3, 0.3)
    line = app.geo_col.add_line(point_sequence=PointSequence(points=[point_one, point_two]))

    point_og = app.geo_col.add_point(0.2, 0.2)
    point = pointer(app, point_og, qtbot)

    line_pen_color_hex_6_digit = convert_rgba_to_hex(line.canvas_item.opts["pen"].color().getRgb())[:-2]
    line_pen_color_setting = matplotlib.colors.cnames[get_setting(f"curve_hovered_pen_color")].lower()
    assert line_pen_color_hex_6_digit == line_pen_color_setting
    app.geo_col.clear_container()
