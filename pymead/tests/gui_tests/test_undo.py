from pytestqt.qtbot import QtBot
from pymead.tests.gui_tests.utils import app, moving_point_2


# def test_undo_two(app, qtbot: QtBot):
#     point_og = app.geo_col.add_point(0.2, 0.6)
#     print(point_og.x())
#     print(point_og.y())
#     point = moving_point_2(app, point_og, qtbot)
#     print(point_og.x())
#     print(point_og.y())
#
#     app.geo_col.clear_container()
