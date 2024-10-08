from pymead.core.param import LengthParam
from pymead.core.point import PointSequence
from pymead.tests.gui_tests.utils import app

import numpy as np


def test_show_hide_objs_point(app):
    point_one = app.geo_col.add_point(0.3, 0.1)
    point_two = app.geo_col.add_point(0.3, 0.1)

    assert point_one.canvas_item.isVisible()
    assert point_two.canvas_item.isVisible()

    app.showHidePymeadObjs("points", show=False)
    assert not point_one.canvas_item.isVisible()
    assert not point_two.canvas_item.isVisible()

    app.showHidePymeadObjs("points", show=True)
    assert point_one.canvas_item.isVisible()
    assert point_two.canvas_item.isVisible()

    app.geo_col.clear_container()


def test_show_hide_objs_lines(app):
    point_one = app.geo_col.add_point(0.2, 0.4)
    point_two = app.geo_col.add_point(0.5, 0.4)
    point_three = app.geo_col.add_point(0.7, 0.2)
    point_four = app.geo_col.add_point(0.1, 0.49)
    line_one = app.geo_col.add_line(point_sequence=PointSequence(points=[point_one, point_two]))
    line_two = app.geo_col.add_line(point_sequence=PointSequence(points=[point_three, point_four]))

    assert line_one.canvas_item.isVisible()
    assert line_two.canvas_item.isVisible()

    app.showHidePymeadObjs("lines", show=False)
    assert not line_one.canvas_item.isVisible()
    assert not line_two.canvas_item.isVisible()

    app.showHidePymeadObjs("lines", show=True)
    assert line_one.canvas_item.isVisible()
    assert line_two.canvas_item.isVisible()

    app.geo_col.clear_container()


def test_show_hide_objs_bezier(app):
    point_one_upper = app.geo_col.add_point(0.0, 0.0)
    point_two_upper = app.geo_col.add_point(0.3, 0.2)
    point_three_upper = app.geo_col.add_point(0.6, 0.15)
    point_four_upper = app.geo_col.add_point(0.8, 0.1)

    point_one_lower = app.geo_col.add_point(0.2, -0.1)
    point_two_lower = app.geo_col.add_point(0.5, 0.0)
    point_three_lower = app.geo_col.add_point(0.8, -0.05)

    bezier_one = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_two_upper,
        point_three_upper,
        point_four_upper
    ]))

    bezier_two = app.geo_col.add_bezier(point_sequence=PointSequence(points=[
        point_one_upper,
        point_one_lower,
        point_two_lower,
        point_three_lower
    ]))

    assert bezier_one.canvas_item.isVisible()
    assert bezier_two.canvas_item.isVisible()

    app.showHidePymeadObjs("bezier", show=False)
    assert not bezier_one.canvas_item.isVisible()
    assert not bezier_two.canvas_item.isVisible()

    app.showHidePymeadObjs("bezier", show=True)
    assert bezier_one.canvas_item.isVisible()
    assert bezier_two.canvas_item.isVisible()

    app.geo_col.clear_container()


def test_show_hide_objs_airfoils(app):
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

    assert airfoil_thin.canvas_item.isVisible()
    assert airfoil_blunt.canvas_item.isVisible()

    app.showHidePymeadObjs("airfoils", show=False)
    assert not airfoil_thin.canvas_item.isVisible()
    assert not airfoil_blunt.canvas_item.isVisible()

    app.showHidePymeadObjs("airfoils", show=True)
    assert airfoil_thin.canvas_item.isVisible()
    assert airfoil_blunt.canvas_item.isVisible()

    app.geo_col.clear_container()


def test_show_hide_objs_geocon(app):
    point_one = app.geo_col.add_point(0.1, 0.3)
    point_two = app.geo_col.add_point(0.1, 0.6)
    length_param = LengthParam(value=0.3, name="DistanceParam", geo_col=app.geo_col)
    geocon_one = app.geo_col.add_constraint(constraint_type="DistanceConstraint",
                                            p1=point_one,
                                            p2=point_two,
                                            value=length_param,
                                            assign_unique_name=True
                                            )

    app.showHidePymeadObjs("geocon", show=False)
    assert geocon_one.canvas_item.isHidden()

    app.showHidePymeadObjs("geocon", show=True)
    assert geocon_one.canvas_item.isShown()
    app.geo_col.clear_container()


def test_show_hide_objs_polylines(app):
    polyline_1 = app.geo_col.add_polyline("n0012-il")
    polyline_2 = app.geo_col.add_polyline("sc20612-il")

    app.showHidePymeadObjs("polylines", show=False)
    assert not polyline_1.canvas_item.isVisible()
    assert not polyline_2.canvas_item.isVisible()

    app.showHidePymeadObjs("polylines", show=True)
    assert polyline_1.canvas_item.isVisible()
    assert polyline_2.canvas_item.isVisible()

    app.geo_col.clear_container()


def test_show_hide_objs_reference(app):
    ref_1 = app.geo_col.add_reference_polyline(source="n0012-il")
    ref_2 = app.geo_col.add_reference_polyline(source="sc20612-il")

    app.showHidePymeadObjs("reference", show=False)
    assert not ref_1.canvas_item.isVisible()
    assert not ref_2.canvas_item.isVisible()

    app.showHidePymeadObjs("reference", show=True)
    assert ref_1.canvas_item.isVisible()
    assert ref_2.canvas_item.isVisible()

    app.geo_col.clear_container()


def test_show_hide_all_objs(app):
    point_one_upper = app.geo_col.add_point(0.0, 0.0)
    point_two_upper = app.geo_col.add_point(0.3, 0.2)
    point_three_upper = app.geo_col.add_point(0.6, 0.15)
    point_four_upper = app.geo_col.add_point(0.8, 0.1)

    point_one_lower = app.geo_col.add_point(0.2, -0.1)
    point_two_lower = app.geo_col.add_point(0.5, 0.0)
    point_three_lower = app.geo_col.add_point(0.8, -0.05)

    midpoint = app.geo_col.add_point(0.8, 0.1)

    length_param = LengthParam(value=0.5, name="DistanceParam", geo_col=app.geo_col)
    geocon_one = app.geo_col.add_constraint(constraint_type="DistanceConstraint",
                                            p1=point_one_upper,
                                            p2=point_two_lower,
                                            value=length_param,
                                            assign_unique_name=True
                                            )

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

    polyline = app.geo_col.add_polyline("n0012-il")
    ref = app.geo_col.add_reference_polyline(source="sc20612-il")

    assert point_four_upper.canvas_item.isVisible()
    assert point_one_lower.canvas_item.isVisible()
    assert bezier_one_blunt.canvas_item.isVisible()
    assert bezier_two_blunt.canvas_item.isVisible()
    assert line_two_blunt.canvas_item.isVisible()
    assert line_one_blunt.canvas_item.isVisible()
    assert airfoil_blunt.canvas_item.isVisible()
    assert geocon_one.canvas_item.isShown()
    assert polyline.canvas_item.isVisible()
    assert ref.canvas_item.isVisible()

    app.hideAllPymeadObjs()
    assert not point_four_upper.canvas_item.isVisible()
    assert not point_one_lower.canvas_item.isVisible()
    assert not bezier_one_blunt.canvas_item.isVisible()
    assert not bezier_two_blunt.canvas_item.isVisible()
    assert not line_two_blunt.canvas_item.isVisible()
    assert not line_one_blunt.canvas_item.isVisible()
    assert not airfoil_blunt.canvas_item.isVisible()
    assert geocon_one.canvas_item.isHidden()
    assert not polyline.canvas_item.isVisible()
    assert not ref.canvas_item.isVisible()

    app.showAllPymeadObjs()
    assert point_four_upper.canvas_item.isVisible()
    assert point_one_lower.canvas_item.isVisible()
    assert bezier_one_blunt.canvas_item.isVisible()
    assert bezier_two_blunt.canvas_item.isVisible()
    assert line_two_blunt.canvas_item.isVisible()
    assert line_one_blunt.canvas_item.isVisible()
    assert airfoil_blunt.canvas_item.isVisible()
    assert geocon_one.canvas_item.isShown()
    assert polyline.canvas_item.isVisible()
    assert ref.canvas_item.isVisible()

    app.geo_col.clear_container()
