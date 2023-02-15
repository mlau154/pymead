def create_bezier_curve_from_ctrlpts(ctrlpt_dict: dict):
    theSession = NXOpen.Session.GetSession()
    workPart = theSession.Parts.Work

    te_lines = []
    te_points = []
    splines = []

    for ctrlpts in ctrlpt_dict.values():
        for coord_set in ctrlpts:
            studioSplineBuilderEx1 = workPart.Features.CreateStudioSplineBuilderEx(NXOpen.NXObject.Null)
            studioSplineBuilderEx1.MatchKnotsType = NXOpen.Features.StudioSplineBuilderEx.MatchKnotsTypes.General
            studioSplineBuilderEx1.Type = NXOpen.Features.StudioSplineBuilderEx.Types.ByPoles
            studioSplineBuilderEx1.IsSingleSegment = True
            for pt in coord_set:
                scalar1 = workPart.Scalars.CreateScalar(pt[0], NXOpen.Scalar.DimensionalityType.NotSet,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)
                scalar2 = workPart.Scalars.CreateScalar(pt[1], NXOpen.Scalar.DimensionalityType.NotSet,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)
                scalar3 = workPart.Scalars.CreateScalar(pt[2], NXOpen.Scalar.DimensionalityType.NotSet,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)
                point = workPart.Points.CreatePoint(scalar1, scalar2, scalar3,
                                                    NXOpen.SmartObject.UpdateOption.WithinModeling)
                point.RemoveViewDependency()
                geometricConstraintData = studioSplineBuilderEx1.ConstraintManager.CreateGeometricConstraintData()
                geometricConstraintData.Point = point
                studioSplineBuilderEx1.ConstraintManager.Append(geometricConstraintData)

            nXObject = studioSplineBuilderEx1.Commit()
            splines.append(nXObject)
            studioSplineBuilderEx1.Destroy()
        te_1 = NXOpen.Point3d(ctrlpts[0][0][0], ctrlpts[0][0][1], ctrlpts[0][0][2])
        te_2 = NXOpen.Point3d(ctrlpts[-1][-1][0], ctrlpts[-1][-1][1], ctrlpts[-1][-1][2])
        te_points.append([te_1, te_2])
        te_lines.append(workPart.Curves.CreateLine(te_1, te_2))


def write_user_expression(expression: str, units: str or None):
    theSession = NXOpen.Session.GetSession()
    workPart = theSession.Parts.Work

    conversions = {"deg": "Degrees", "mm": "MilliMeter", "rad": "Radians"}

    if isinstance(units, str):
        if units in conversions.keys():
            converted_unit = conversions[units]
            workPart.Expressions.CreateNumberExpression(expression, workPart.UnitCollection.FindObject(converted_unit))
        else:
            raise ValueError(f"Unit {units} not found in conversions dictionary ({conversions})")
    elif units is None:
        converted_unit = NXOpen.Unit.Null
        workPart.Expressions.CreateNumberExpression(expression, converted_unit)
    else:
        raise TypeError(
            f"The data type of \"units\" does not match a recognized data type "
            f"(input type: {type(units)}, accepted types: {str} or {None})")


def add_sketch(airfoil_data: dict):
    theSession = NXOpen.Session.GetSession()
    workPart = theSession.Parts.Work
    displayPart = theSession.Parts.Display
    # ----------------------------------------------
    #   Menu: Insert->Sketch
    # ----------------------------------------------

    sketchInPlaceBuilder1 = workPart.Sketches.CreateSketchInPlaceBuilder2(NXOpen.Sketch.Null)

    origin1 = NXOpen.Point3d(0.0, 0.0, 0.0)
    normal1 = NXOpen.Vector3d(0.0, 0.0, 1.0)
    plane1 = workPart.Planes.CreatePlane(origin1, normal1, NXOpen.SmartObject.UpdateOption.WithinModeling)

    sketchInPlaceBuilder1.PlaneReference = plane1

    # unit1 = workPart.UnitCollection.FindObject("MilliMeter")
    # expression1 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)
    #
    # expression2 = workPart.Expressions.CreateSystemExpressionWithUnits("0", unit1)

    sketchAlongPathBuilder1 = workPart.Sketches.CreateSketchAlongPathBuilder(NXOpen.Sketch.Null)

    simpleSketchInPlaceBuilder1 = workPart.Sketches.CreateSimpleSketchInPlaceBuilder()

    sketchAlongPathBuilder1.PlaneLocation.Expression.SetFormula("0")

    # theSession.SetUndoMarkName(markId3, "Create Sketch Dialog")

    simpleSketchInPlaceBuilder1.UseWorkPartOrigin = False

    coordinates1 = NXOpen.Point3d(0.0, 0.0, 0.0)
    point1 = workPart.Points.CreatePoint(coordinates1)

    origin2 = NXOpen.Point3d(0.0, 0.0, 0.0)
    matrix1 = NXOpen.Matrix3x3()

    matrix1.Xx = 1.0
    matrix1.Xy = 0.0
    matrix1.Xz = 0.0
    matrix1.Yx = -0.0
    matrix1.Yy = 0.0
    matrix1.Yz = 1.0
    matrix1.Zx = 0.0
    matrix1.Zy = -1.0
    matrix1.Zz = 0.0
    plane2 = workPart.Planes.CreateFixedTypePlane(origin2, matrix1, NXOpen.SmartObject.UpdateOption.WithinModeling)

    coordinates2 = NXOpen.Point3d(0.0, 0.0, 0.0)
    point2 = workPart.Points.CreatePoint(coordinates2)

    origin3 = NXOpen.Point3d(0.0, 0.0, 0.0)
    vector1 = NXOpen.Vector3d(0.0, -1.0, 0.0)
    direction1 = workPart.Directions.CreateDirection(origin3, vector1, NXOpen.SmartObject.UpdateOption.WithinModeling)

    origin4 = NXOpen.Point3d(0.0, 0.0, 0.0)
    vector2 = NXOpen.Vector3d(1.0, 0.0, 0.0)
    direction2 = workPart.Directions.CreateDirection(origin4, vector2, NXOpen.SmartObject.UpdateOption.WithinModeling)

    origin5 = NXOpen.Point3d(0.0, 0.0, 0.0)
    matrix2 = NXOpen.Matrix3x3()

    matrix2.Xx = 1.0
    matrix2.Xy = 0.0
    matrix2.Xz = 0.0
    matrix2.Yx = -0.0
    matrix2.Yy = 0.0
    matrix2.Yz = 1.0
    matrix2.Zx = 0.0
    matrix2.Zy = -1.0
    matrix2.Zz = 0.0
    plane3 = workPart.Planes.CreateFixedTypePlane(origin5, matrix2, NXOpen.SmartObject.UpdateOption.WithinModeling)

    xform1 = workPart.Xforms.CreateXformByPlaneXDirPoint(plane3, direction2, point2,
                                                         NXOpen.SmartObject.UpdateOption.WithinModeling, 0.625, False,
                                                         False)

    cartesianCoordinateSystem1 = workPart.CoordinateSystems.CreateCoordinateSystem(xform1,
                                                                                   NXOpen.SmartObject.UpdateOption.WithinModeling)

    simpleSketchInPlaceBuilder1.CoordinateSystem = cartesianCoordinateSystem1

    datumAxis1 = workPart.Datums.FindObject("DATUM_CSYS(0) X axis")
    simpleSketchInPlaceBuilder1.HorizontalReference.Value = datumAxis1

    point3 = simpleSketchInPlaceBuilder1.SketchOrigin

    simpleSketchInPlaceBuilder1.SketchOrigin = point3

    markId4 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Create Sketch")

    theSession.DeleteUndoMark(markId4, None)

    markId5 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Create Sketch")

    theSession.Preferences.Sketch.CreateInferredConstraints = False

    theSession.Preferences.Sketch.ContinuousAutoDimensioning = False

    theSession.Preferences.Sketch.DimensionLabel = NXOpen.Preferences.SketchPreferences.DimensionLabelType.Expression

    theSession.Preferences.Sketch.TextSizeFixed = False

    theSession.Preferences.Sketch.CreatePersistentRelations = False

    theSession.Preferences.Sketch.FixedTextSize = 3.0

    theSession.Preferences.Sketch.DisplayParenthesesOnReferenceDimensions = True

    theSession.Preferences.Sketch.DisplayReferenceGeometry = False

    theSession.Preferences.Sketch.DisplayShadedRegions = True

    theSession.Preferences.Sketch.FindMovableObjects = True

    theSession.Preferences.Sketch.ConstraintSymbolSize = 3.0

    theSession.Preferences.Sketch.DisplayObjectColor = False

    theSession.Preferences.Sketch.DisplayObjectName = True

    theSession.Preferences.Sketch.EditDimensionOnCreation = True

    theSession.Preferences.Sketch.CreateDimensionForTypedValues = True

    theSession.Preferences.Sketch.ScaleOnFirstDrivingDimension = False

    nXObject1 = simpleSketchInPlaceBuilder1.Commit()

    sketch1 = nXObject1
    feature1 = sketch1.Feature

    markId6 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "update")

    nErrs1 = theSession.UpdateManager.DoUpdate(markId6)

    sketch1.Activate(NXOpen.Sketch.ViewReorient.TrueValue)

    theSession.Preferences.Sketch.FindMovableObjects = True

    sketchFindMovableObjectsBuilder1 = workPart.Sketches.CreateFindMovableObjectsBuilder()

    nXObject2 = sketchFindMovableObjectsBuilder1.Commit()

    sketchFindMovableObjectsBuilder1.Destroy()

    theSession.DeleteUndoMark(markId5, None)

    # theSession.SetUndoMarkName(markId3, "Create Sketch")

    sketchInPlaceBuilder1.Destroy()

    sketchAlongPathBuilder1.Destroy()

    simpleSketchInPlaceBuilder1.Destroy()

    # plane1.DestroyPlane()

    theSession.ActiveSketch.SetName("Airfoil_Sketch_001")

    # ----------------------------------------------
    #   Menu: Task->Finish Sketch
    # ----------------------------------------------
    sketchWorkRegionBuilder1 = workPart.Sketches.CreateWorkRegionBuilder()

    sketchWorkRegionBuilder1.Scope = NXOpen.SketchWorkRegionBuilder.ScopeType.EntireSketch

    nXObject3 = sketchWorkRegionBuilder1.Commit()

    sketchWorkRegionBuilder1.Destroy()

    theSession.ActiveSketch.CalculateStatus()

    theSession.Preferences.Sketch.SectionView = False

    for airfoil_name, airfoil in airfoil_data.items():
        points = airfoil['control_point_list_string']
        point_list = [NXOpen.Point3d(point[0], point[1], point[2]) for point in points]
        lines = [workPart.Curves.CreateLine(point_list[idx], point_list[idx + 1]) for idx in range(len(point_list) - 1)]

        le_point = NXOpen.Point3d(airfoil['dx'], 0.0, airfoil['dy'])
        te_point = NXOpen.Point3d(airfoil['dx'] + airfoil['c'] * math.cos(-airfoil['alf']),
                                   0.0,
                                   airfoil['dy'] + airfoil['c'] * math.sin(-airfoil['alf']))

        chord_line = workPart.Curves.CreateLine(le_point, te_point)
        te_thickness_upper_line = workPart.Curves.CreateLine(te_point, point_list[0])
        te_thickness_lower_line = workPart.Curves.CreateLine(te_point, point_list[-1])
        lines.insert(0, te_thickness_upper_line)
        lines.insert(0, chord_line)
        lines.append(te_thickness_lower_line)

        airfoil_data[airfoil_name]['lines'] = lines

        convertToFromReferenceBuilder1 = workPart.Sketches.CreateConvertToFromReferenceBuilder()
        convert_lines_to_reference_list = convertToFromReferenceBuilder1.InputObjects

        for idx, line in enumerate(lines):
            theSession.ActiveSketch.AddGeometry(line, NXOpen.Sketch.InferConstraintsOption.InferNoConstraints)
            convert_lines_to_reference_list.Add(line)

        convertToFromReferenceBuilder1.OutputState = NXOpen.ConvertToFromReferenceBuilder.OutputType.Reference
        convertToFromReferenceBuilder1.Commit()
        convertToFromReferenceBuilder1.Destroy()

    # -------------------------------------------

    # ----------------------------------------------
    #   Menu: Insert->Curve->Spline...
    # ----------------------------------------------

    def generate_splines(curve_orders):
        starting_point = 2
        for curve_order in curve_orders:
            sketchSplineBuilder = workPart.Features.CreateSketchSplineBuilder(NXOpen.Spline.Null)

            # # origin1 = NXOpen.Point3d(0.0, 0.0, 0.0)
            # # normal1 = NXOpen.Vector3d(0.0, 0.0, 1.0)
            # # plane1 = workPart.Planes.CreatePlane(origin1, normal1, NXOpen.SmartObject.UpdateOption.WithinModeling)
            #
            # sketchSplineBuilder.DrawingPlane = plane1
            #
            # unit1 = sketchSplineBuilder.Extender.StartValue.Units

            # origin2 = NXOpen.Point3d(0.0, 0.0, 0.0)
            # normal2 = NXOpen.Vector3d(0.0, 0.0, 1.0)
            # plane2 = workPart.Planes.CreatePlane(origin2, normal2, NXOpen.SmartObject.UpdateOption.WithinModeling)
            #
            # sketchSplineBuilder.MovementPlane = plane1
            #
            # sketchSplineBuilder.OrientExpress.ReferenceOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Reference.WcsDisplayPart
            #
            # sketchSplineBuilder.MovementMethod = NXOpen.Features.StudioSplineBuilderEx.MovementMethodType.WCS
            #
            # sketchSplineBuilder.CanUseOrientationTool = True
            #
            # sketchSplineBuilder.WCSOption = NXOpen.Features.StudioSplineBuilderEx.WCSOptionType.Z
            #
            sketchSplineBuilder.Type = NXOpen.Features.StudioSplineBuilderEx.Types.ByPoles  # Rather than through points
            #
            sketchSplineBuilder.IsSingleSegment = True  # Bézier curve rather than B-spline/NURBS
            #
            # sketchSplineBuilder.OrientExpress.AxisOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Axis.Z
            #
            # sketchSplineBuilder.OrientExpress.PlaneOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Plane.Passive
            #
            # sketchSplineBuilder.Extender.StartValue.SetFormula("0")
            #
            # sketchSplineBuilder.Extender.EndValue.SetFormula("0")
            #
            # sketchSplineBuilder.InputCurveOption = NXOpen.Features.StudioSplineBuilderEx.InputCurveOptions.Hide
            #
            # sketchSplineBuilder.MatchKnotsType = NXOpen.Features.StudioSplineBuilderEx.MatchKnotsTypes.NotSet
            #
            # sketchSplineBuilder.OrientExpress.AxisOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Axis.Passive
            #
            # sketchSplineBuilder.OrientExpress.AxisOption = NXOpen.GeometricUtilities.OrientXpressBuilder.Axis.Z
            #
            # sketchSplineBuilder.IsAssociative = False

            lines_in_curve = lines[starting_point:starting_point + curve_order]

            listing_window = theSession.ListingWindow

            listing_window.Open()

            for idx, curve_line in enumerate(lines_in_curve):
                if idx == 0:
                    scalar = workPart.Scalars.CreateScalar(0.0, NXOpen.Scalar.DimensionalityType.NotSet,
                                                           NXOpen.SmartObject.UpdateOption.WithinModeling)

                    point = workPart.Points.CreatePoint(curve_line, scalar,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)

                    point.RemoveViewDependency()
                    # point_3d = point.Coordinates
                    #
                    # listing_window.WriteFullline(f"point coordinates = {point_3d.X}, {point_3d.Y}, {point_3d.Z}")

                    geometricConstraintData = sketchSplineBuilder.ConstraintManager.CreateGeometricConstraintData()

                    geometricConstraintData.Point = point

                    sketchSplineBuilder.ConstraintManager.Append(geometricConstraintData)

                    # sketchSplineBuilder.Evaluate()

                    scalar = workPart.Scalars.CreateScalar(1.0, NXOpen.Scalar.DimensionalityType.NotSet,
                                                           NXOpen.SmartObject.UpdateOption.WithinModeling)

                    point = workPart.Points.CreatePoint(curve_line, scalar,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)

                    point.RemoveViewDependency()
                    # point_3d = point.Coordinates
                    #
                    # listing_window.WriteFullline(f"point coordinates = {point_3d.X}, {point_3d.Y}, {point_3d.Z}")

                    geometricConstraintData = sketchSplineBuilder.ConstraintManager.CreateGeometricConstraintData()

                    geometricConstraintData.Point = point

                    sketchSplineBuilder.ConstraintManager.Append(geometricConstraintData)

                    # sketchSplineBuilder.Evaluate()

                else:
                    scalar = workPart.Scalars.CreateScalar(1.0, NXOpen.Scalar.DimensionalityType.NotSet,
                                                           NXOpen.SmartObject.UpdateOption.WithinModeling)

                    point = workPart.Points.CreatePoint(curve_line, scalar,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)

                    point.RemoveViewDependency()
                    # point_3d = point.Coordinates
                    #
                    # listing_window.WriteFullline(f"point coordinates = {point_3d.X}, {point_3d.Y}, {point_3d.Z}")

                    geometricConstraintData = sketchSplineBuilder.ConstraintManager.CreateGeometricConstraintData()

                    geometricConstraintData.Point = point

                    sketchSplineBuilder.ConstraintManager.Append(geometricConstraintData)

                    # sketchSplineBuilder.Evaluate()

            sketchSplineBuilder.Evaluate()

            listing_window.Close()

            sketchSplineBuilder.Commit()

            # listing_window = theSession.ListingWindow
            #
            # listing_window.Open()
            # listing_window.WriteFullline(f"type = {type(sketchSplineBuilder.Curve)}")
            # listing_window.WriteFullline(f"type of lines[0] = {type(lines[0])}")
            # listing_window.WriteFullline(f"type of lines[1] = {type(lines[1])}")
            # poles = sketchSplineBuilder.Curve.Get3DPoles()
            # for idx, pole in enumerate(poles):
            #     listing_window.WriteFullline(f"pole {idx} X, Y, Z = {pole.X}, {pole.Y}, {pole.Z}")
            # listing_window.WriteFullline(f"poles = {sketchSplineBuilder.Curve.Get3DPoles()}")
            # listing_window.Close()

            spline1 = sketchSplineBuilder.Curve
            for idx, curve_line in enumerate(lines_in_curve):
                if idx == 0:
                    geom1 = NXOpen.Sketch.ConstraintGeometry()
                    geom1.Geometry = spline1
                    geom1.PointType = NXOpen.Sketch.ConstraintPointType.StartVertex
                    geom1.SplineDefiningPointIndex = 0
                    geom2 = NXOpen.Sketch.ConstraintGeometry()
                    geom2.Geometry = curve_line
                    geom2.PointType = NXOpen.Sketch.ConstraintPointType.StartVertex
                    theSession.ActiveSketch.CreateCoincidentConstraint(geom1, geom2)

                    geom1 = NXOpen.Sketch.ConstraintGeometry()
                    geom1.Geometry = spline1
                    geom1.PointType = NXOpen.Sketch.ConstraintPointType.SplinePole
                    geom1.SplineDefiningPointIndex = 1
                    geom2 = NXOpen.Sketch.ConstraintGeometry()
                    geom2.Geometry = curve_line
                    geom2.PointType = NXOpen.Sketch.ConstraintPointType.EndVertex
                    theSession.ActiveSketch.CreateCoincidentConstraint(geom1, geom2)
                else:
                    geom1 = NXOpen.Sketch.ConstraintGeometry()
                    geom1.Geometry = spline1
                    geom1.PointType = NXOpen.Sketch.ConstraintPointType.SplinePole
                    geom1.SplineDefiningPointIndex = idx + 1
                    geom2 = NXOpen.Sketch.ConstraintGeometry()
                    geom2.Geometry = curve_line
                    geom2.PointType = NXOpen.Sketch.ConstraintPointType.EndVertex
                    theSession.ActiveSketch.CreateCoincidentConstraint(geom1, geom2)

            theSession.ActiveSketch.Update()

            # sketchFindMovableObjectsBuilder2 = workPart.Sketches.CreateFindMovableObjectsBuilder()
            #
            # nXObject3 = sketchFindMovableObjectsBuilder2.Commit()
            #
            # sketchFindMovableObjectsBuilder2.Destroy()

            sketchSplineBuilder.Destroy()

            starting_point += curve_order
            # -------------------------------------------

    for airfoil_name, airfoil in airfoil_data.items():
        generate_splines(airfoil['curve_orders'])

    # # ================================= ADD ANGULAR DIMENSION =======================================

    lines = airfoil_data['A0']['lines']

    for airfoil_name, airfoil in airfoil_data.items():
        for idx, line in enumerate(airfoil['lines']):
            if any(substr in airfoil_data[airfoil_name]['line_tags'][idx] for substr in ['theta_te_upper',
                                                                                         'theta_te_lower', 'psi1_le',
                                                                                         'psi2_le', 'le_angle_180']):
                if any(substr in airfoil_data[airfoil_name]['line_tags'][idx] for substr in ['theta_te_upper',
                                                                                         'theta_te_lower']):
                    line1 = line
                    line2 = airfoil['lines'][0]
                else:
                    line1 = line
                    line2 = airfoil['lines'][idx + 1]
                sketchAngularDimensionBuilder2 = workPart.Sketches.CreateAngularDimensionBuilder(
                    NXOpen.Annotations.AngularDimension.Null)

                sketchAngularDimensionBuilder2.Driving.DrivingMethod = NXOpen.Annotations.DrivingValueBuilder.DrivingValueMethod.Driving

                # sketchAngularDimensionBuilder2.Driving.DimensionValue = 139.0

                scalar1 = workPart.Scalars.CreateScalar(0.0, NXOpen.Scalar.DimensionalityType.NotSet,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)
                point1 = workPart.Points.CreatePoint(line1, scalar1, NXOpen.SmartObject.UpdateOption.WithinModeling)
                point1_3d = point1.Coordinates
                # point1_3d = NXOpen.Point3d(0.0, 0.0, 0.0)

                sketchAngularDimensionBuilder2.FirstAssociativity.SetValue(line1, workPart.ModelingViews.WorkView, point1_3d)

                scalar2 = workPart.Scalars.CreateScalar(1.0, NXOpen.Scalar.DimensionalityType.NotSet,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)
                point2 = workPart.Points.CreatePoint(line2, scalar2, NXOpen.SmartObject.UpdateOption.WithinModeling)

                point2_3d = point2.Coordinates
                # point2_3d = NXOpen.Point3d(0.5, 0.0, 0.0)

                sketchAngularDimensionBuilder2.SecondAssociativity.SetValue(line2, workPart.ModelingViews.WorkView, point2_3d)

                sketchAngularDimensionBuilder2.Commit()

                sketchAngularDimensionBuilder2.Destroy()

                # # Add second angular dimension: -------------------------------------------------------------------
                #
                # sketchAngularDimensionBuilder2 = workPart.Sketches.CreateAngularDimensionBuilder(
                #     NXOpen.Annotations.AngularDimension.Null)
                #
                # sketchAngularDimensionBuilder2.Driving.DrivingMethod = NXOpen.Annotations.DrivingValueBuilder.DrivingValueMethod.Driving
                #
                # scalar1 = workPart.Scalars.CreateScalar(0.0, NXOpen.Scalar.DimensionalityType.NotSet,
                #                                         NXOpen.SmartObject.UpdateOption.WithinModeling)
                # point1 = workPart.Points.CreatePoint(lines[3], scalar1, NXOpen.SmartObject.UpdateOption.WithinModeling)
                # point1_3d = point1.Coordinates
                # # point1_3d = NXOpen.Point3d(0.0, 0.0, 0.0)
                #
                # sketchAngularDimensionBuilder2.FirstAssociativity.SetValue(lines[3], workPart.ModelingViews.WorkView, point1_3d)
                #
                # scalar2 = workPart.Scalars.CreateScalar(1.0, NXOpen.Scalar.DimensionalityType.NotSet,
                #                                         NXOpen.SmartObject.UpdateOption.WithinModeling)
                # point2 = workPart.Points.CreatePoint(lines[4], scalar2, NXOpen.SmartObject.UpdateOption.WithinModeling)
                #
                # point2_3d = point2.Coordinates
                # # point2_3d = NXOpen.Point3d(0.5, 0.0, 0.0)
                #
                # sketchAngularDimensionBuilder2.SecondAssociativity.SetValue(lines[4], workPart.ModelingViews.WorkView, point2_3d)
                #
                # sketchAngularDimensionBuilder2.Commit()
                #
                # sketchAngularDimensionBuilder2.Destroy()

    # =============================== END ADD ANGULAR DIMENSION =====================================

    # =============================== BEGIN ADD LINEAR DIMENSION ====================================

    for airfoil_name, airfoil in airfoil_data.items():
        for idx, line in enumerate(airfoil['lines']):
            if 'pt2pt_linear' in airfoil_data[airfoil_name]['line_tags'][idx]:
                #
                # if idx == 1:
                #     continue

                sketchLinearDimensionBuilder = workPart.Sketches.CreateLinearDimensionBuilder(NXOpen.Annotations.Dimension.Null)

                # lines9 = []
                # sketchLinearDimensionBuilder.AppendedText.SetBefore(lines9)
                #
                # lines10 = []
                # sketchLinearDimensionBuilder.AppendedText.SetAfter(lines10)
                #
                # lines11 = []
                # sketchLinearDimensionBuilder.AppendedText.SetAbove(lines11)
                #
                # lines12 = []
                # sketchLinearDimensionBuilder.AppendedText.SetBelow(lines12)

                # sketchLinearDimensionBuilder.Origin.SetInferRelativeToGeometry(True)

                sketchLinearDimensionBuilder.Measurement.Method = NXOpen.Annotations.DimensionMeasurementBuilder.MeasurementMethod.PointToPoint

                # sketchLinearDimensionBuilder.Style.DimensionStyle.LimitFitDeviation = "H"
                #
                # sketchLinearDimensionBuilder.Style.DimensionStyle.LimitFitShaftDeviation = "g"

                # sketchLinearDimensionBuilder.Driving.DimensionValue = 3.3333333333000001

                # theSession.SetUndoMarkName(markId6, "Linear Dimension Dialog")
                #
                # sketchLinearDimensionBuilder.Origin.Plane.PlaneMethod = NXOpen.Annotations.PlaneBuilder.PlaneMethodType.XyPlane
                #
                # sketchLinearDimensionBuilder.Origin.SetInferRelativeToGeometry(True)

                sketchLinearDimensionBuilder.Driving.DrivingMethod = NXOpen.Annotations.DrivingValueBuilder.DrivingValueMethod.Driving

                # sketchLinearDimensionBuilder.Origin.Anchor = NXOpen.Annotations.OriginBuilder.AlignmentPosition.MidCenter
                #
                # sketchLinearDimensionBuilder.Origin.SetInferRelativeToGeometry(False)
                #
                # sketchLinearDimensionBuilder.Origin.SetInferRelativeToGeometry(False)
                #
                # sketchLinearDimensionBuilder.Measurement.Direction = NXOpen.Direction.Null
                #
                # sketchLinearDimensionBuilder.Measurement.DirectionView = NXOpen.View.Null
                #
                # sketchLinearDimensionBuilder.Origin.SetInferRelativeToGeometry(False)
                #
                # sketchLinearDimensionBuilder.Style.DimensionStyle.NarrowDisplayType = NXOpen.Annotations.NarrowDisplayOption.NotSet
                #
                # sketchLinearDimensionBuilder.Driving.DrivingMethod = NXOpen.Annotations.DrivingValueBuilder.DrivingValueMethod.Driving

                sketchLinearDimensionBuilder.Origin.SetInferRelativeToGeometry(True)

                # scaleAboutPoint11 = NXOpen.Point3d(-30.83415263885805, -7.9847398527335898, 0.0)
                # viewCenter11 = NXOpen.Point3d(30.83415263885805, 7.9847398527336049, 0.0)
                # workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint11, viewCenter11)
                #
                # scaleAboutPoint12 = NXOpen.Point3d(-24.667322111086438, -6.3877918821868729, 0.0)
                # viewCenter12 = NXOpen.Point3d(24.667322111086438, 6.3877918821868853, 0.0)
                # workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint12, viewCenter12)
                #
                # scaleAboutPoint13 = NXOpen.Point3d(-19.733857688869154, -5.1102335057494974, 0.0)
                # viewCenter13 = NXOpen.Point3d(19.733857688869154, 5.1102335057495125, 0.0)
                # workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint13, viewCenter13)
                #
                # scaleAboutPoint14 = NXOpen.Point3d(-15.787086151095325, -4.0881868045995908, 0.0)
                # viewCenter14 = NXOpen.Point3d(15.787086151095325, 4.0881868045996104, 0.0)
                # workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint14, viewCenter14)
                #
                # scaleAboutPoint15 = NXOpen.Point3d(-12.629668920876258, -3.2705494436796698, 0.0)
                # viewCenter15 = NXOpen.Point3d(12.629668920876258, 3.270549443679692, 0.0)
                # workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint15, viewCenter15)
                #
                # scaleAboutPoint16 = NXOpen.Point3d(-10.103735136701006, -2.6164395549437351, 0.0)
                # viewCenter16 = NXOpen.Point3d(10.103735136701006, 2.6164395549437556, 0.0)
                # workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint16, viewCenter16)
                #
                # origin3 = NXOpen.Point3d(4.8998729536255894, 0.0, -6.8372906550088608)
                # workPart.ModelingViews.WorkView.SetOrigin(origin3)
                #
                # origin4 = NXOpen.Point3d(4.8998729536255894, 0.0, -6.8372906550088608)
                # workPart.ModelingViews.WorkView.SetOrigin(origin4)

                # line2 = theSession.ActiveSketch.FindObject("Curve Line6")
                # point1_5 = NXOpen.Point3d(-5.4636959873285264e-16, 0.0, -4.9999999999999858)
                # point2_5 = NXOpen.Point3d(0.0, 0.0, 0.0)
                # sketchLinearDimensionBuilder.FirstAssociativity.SetValue(NXOpen.InferSnapType.SnapType.Start, line2,
                #                                                           workPart.ModelingViews.WorkView, point1_5,
                #                                                           NXOpen.TaggedObject.Null, NXOpen.View.Null, point2_5)
                #
                # sketchLinearDimensionBuilder.Origin.SetInferRelativeToGeometry(True)
                #
                # point1_6 = NXOpen.Point3d(3.333333333333397, 0.0, -5.0000000000000044)
                # point2_6 = NXOpen.Point3d(0.0, 0.0, 0.0)
                # sketchLinearDimensionBuilder.SecondAssociativity.SetValue(NXOpen.InferSnapType.SnapType.End, line2,
                #                                                            workPart.ModelingViews.WorkView, point1_6,
                #                                                            NXOpen.TaggedObject.Null, NXOpen.View.Null, point2_6)

                scalar1 = workPart.Scalars.CreateScalar(0.0, NXOpen.Scalar.DimensionalityType.NotSet,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)
                point1 = workPart.Points.CreatePoint(line, scalar1, NXOpen.SmartObject.UpdateOption.WithinModeling)
                point1_3d = point1.Coordinates
                # point1_3d = NXOpen.Point3d(0.0, 0.0, 0.0)

                # sketchAngularDimensionBuilder2.FirstAssociativity.SetValue(lines[3], workPart.ModelingViews.WorkView, point1_3d)

                # point1_7 = NXOpen.Point3d(-5.4636959873285264e-16, 0.0, -4.9999999999999858)
                point2_7 = NXOpen.Point3d(0.0, 0.0, 0.0)
                sketchLinearDimensionBuilder.FirstAssociativity.SetValue(NXOpen.InferSnapType.SnapType.Start, line,
                                                                          workPart.ModelingViews.WorkView, point1_3d,
                                                                          NXOpen.TaggedObject.Null, NXOpen.View.Null, point2_7)
                # sketchLinearDimensionBuilder.FirstAssociativity.SetValue(lines[2], workPart.ModelingViews.WorkView, point1_3d)

                scalar2 = workPart.Scalars.CreateScalar(1.0, NXOpen.Scalar.DimensionalityType.NotSet,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)
                point2 = workPart.Points.CreatePoint(line, scalar2, NXOpen.SmartObject.UpdateOption.WithinModeling)
                point2_3d = point2.Coordinates

                # point1_8 = NXOpen.Point3d(3.333333333333397, 0.0, -5.0000000000000044)
                point2_8 = NXOpen.Point3d(0.0, 0.0, 0.0)
                sketchLinearDimensionBuilder.SecondAssociativity.SetValue(NXOpen.InferSnapType.SnapType.End, line,
                                                                           workPart.ModelingViews.WorkView, point2_3d,
                                                                           NXOpen.TaggedObject.Null, NXOpen.View.Null, point2_8)

                # sketchLinearDimensionBuilder.SecondAssociativity.SetValue(lines[2], workPart.ModelingViews.WorkView, point2_3d)

                sketchLinearDimensionBuilder.Origin.SetInferRelativeToGeometry(True)

                # scaleAboutPoint17 = NXOpen.Point3d(-1.0268291083552799, -6.3584417863538469, 0.0)
                # viewCenter17 = NXOpen.Point3d(1.0268291083552799, 6.3584417863538709, 0.0)
                # workPart.ModelingViews.WorkView.ZoomAboutPoint(0.80000000000000004, scaleAboutPoint17, viewCenter17)

                # basePart9 = theSession.Parts.BaseWork

                # sketchLinearDimensionBuilder.Origin.SetInferRelativeToGeometryFromLeader(True)

                # assocOrigin2 = NXOpen.Annotations.Annotation.AssociativeOriginData()
                #
                # assocOrigin2.OriginType = NXOpen.Annotations.AssociativeOriginType.RelativeToGeometry
                # assocOrigin2.View = NXOpen.View.Null
                # assocOrigin2.ViewOfGeometry = workPart.ModelingViews.WorkView
                # point3 = workPart.Points.FindObject("ENTITY 2 2")
                # assocOrigin2.PointOnGeometry = point3
                # assocOrigin2.VertAnnotation = NXOpen.Annotations.Annotation.Null
                # assocOrigin2.VertAlignmentPosition = NXOpen.Annotations.AlignmentPosition.TopLeft
                # assocOrigin2.HorizAnnotation = NXOpen.Annotations.Annotation.Null
                # assocOrigin2.HorizAlignmentPosition = NXOpen.Annotations.AlignmentPosition.TopLeft
                # assocOrigin2.AlignedAnnotation = NXOpen.Annotations.Annotation.Null
                # assocOrigin2.DimensionLine = 0
                # assocOrigin2.AssociatedView = NXOpen.View.Null
                # assocOrigin2.AssociatedPoint = NXOpen.Point.Null
                # assocOrigin2.OffsetAnnotation = NXOpen.Annotations.Annotation.Null
                # assocOrigin2.OffsetAlignmentPosition = NXOpen.Annotations.AlignmentPosition.TopLeft
                # assocOrigin2.XOffsetFactor = 0.0
                # assocOrigin2.YOffsetFactor = 0.0
                # assocOrigin2.StackAlignmentPosition = NXOpen.Annotations.StackAlignmentPosition.Above
                # sketchLinearDimensionBuilder.Origin.SetAssociativeOrigin(assocOrigin2)

                # point4 = NXOpen.Point3d(0.97685918067848476, 0.0, -8.7527218763638981)
                # sketchLinearDimensionBuilder.Origin.Origin.SetValue(NXOpen.TaggedObject.Null, NXOpen.View.Null, point4)
                #
                # sketchLinearDimensionBuilder.Origin.SetInferRelativeToGeometry(True)

                # sketchLinearDimensionBuilder.Style.LineArrowStyle.LeaderOrientation = NXOpen.Annotations.LeaderSide.Right
                #
                # sketchLinearDimensionBuilder.Style.DimensionStyle.TextCentered = True

                # markId7 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Linear Dimension")

                sketchLinearDimensionBuilder.Commit()

                # sketchFindMovableObjectsBuilder3 = workPart.Sketches.CreateFindMovableObjectsBuilder()
                #
                # sketchFindMovableObjectsBuilder3.Commit()

                # sketchFindMovableObjectsBuilder3.Destroy()

                sketchLinearDimensionBuilder.Destroy()

    # =============================== END ADD LINEAR DIMENSION ======================================

    theSession.ActiveSketch.Update()

    theSession.ActiveSketch.Deactivate(NXOpen.Sketch.ViewReorient.TrueValue, NXOpen.Sketch.UpdateLevel.Model)

    # expression1 = workPart.Expressions.FindObject("p7")
    # unit1 = workPart.UnitCollection.FindObject("Inch")
    # workPart.Expressions.EditExpressionWithUnits(expression1, unit1, "A0_Base_L2_te * A0_Base_c")
    # expression1 = workPart.Expressions.FindObject("p2")
    # workPart.Expressions.EditExpressionWithUnits(expression1, unit1, "A0_Base_L1_te * A0_Base_c")
    #
    # markId4 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "NX update")
    #
    # nErrs1 = theSession.UpdateManager.DoUpdate(markId4)


def create_connected_lines_from_points(points: list):
    theSession = NXOpen.Session.GetSession()
    workPart = theSession.Parts.Work
    displayPart = theSession.Parts.Display
    basePart1 = theSession.Parts.BaseWork

    # # markId1 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Edit Sketch")
    #
    # # theSession.BeginTaskEnvironment()
    #
    # # markId2 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Enter Sketch")
    #
    # theUI = NXOpen.UI.GetUI()
    #
    # # theUI.SelectionManager.GetSelectedTaggedObject(0).Activate(NXOpen.Sketch.ViewReorient.TrueValue)
    #
    # theSession.Preferences.Sketch.FindMovableObjects = True
    #
    # sketchFindMovableObjectsBuilder1 = workPart.Sketches.CreateFindMovableObjectsBuilder()
    #
    # nXObject1 = sketchFindMovableObjectsBuilder1.Commit()
    #
    # sketchFindMovableObjectsBuilder1.Destroy()
    #
    # # theSession.DeleteUndoMarksUpToMark(markId2, None, True)
    #
    # # markId3 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Open Sketch")
    #
    # basePart2 = theSession.Parts.BaseWork
    #
    # basePart3 = theSession.Parts.BaseWork

    # scaleAboutPoint1 = NXOpen.Point3d(22.418806260421853, 8.2505759080233005, 0.0)
    # viewCenter1 = NXOpen.Point3d(-22.418806260421906, -8.2505759080232757, 0.0)
    # workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint1, viewCenter1)
    #
    # scaleAboutPoint2 = NXOpen.Point3d(17.935045008337472, 6.6004607264186417, 0.0)
    # viewCenter2 = NXOpen.Point3d(-17.935045008337539, -6.6004607264186141, 0.0)
    # workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint2, viewCenter2)
    #
    # scaleAboutPoint3 = NXOpen.Point3d(14.348036006669966, 5.2803685811349137, 0.0)
    # viewCenter3 = NXOpen.Point3d(-14.348036006670039, -5.2803685811348862, 0.0)
    # workPart.ModelingViews.WorkView.ZoomAboutPoint(1.25, scaleAboutPoint3, viewCenter3)

    # ----------------------------------------------
    #   Menu: Insert->Curve->Line...
    # ----------------------------------------------
    point_list = [NXOpen.Point3d(point[0], point[1], point[2]) for point in points]
    # lines = [workPart.Curves.CreateLine(point_list[idx], point_list[idx + 1]) for idx in range(len(point_list) - 1)]

    for idx in range(len(point_list) - 1):
        lineNone = NXOpen.Features.AssociativeLine()
        # listing_window = theSession.ListingWindow
        #
        # listing_window.Open()
        # listing_window.WriteFullline(f"type = {type(lineNone)}")
        # listing_window.Close()
        builder = NXOpen.Features.BaseFeatureCollection.CreateAssociativeLineBuilder(lineNone)
        builder.Associative = True
        builder.StartPointOptions = NXOpen.Features.AssociativeLineBuilder.StartOption.Point
        builder.StartPoint.Value = point_list[idx]
        builder.EndPointOptions = NXOpen.Features.AssociativeLineBuilder.EndOption.Point
        builder.EndPoint.Value = point_list[idx + 1]
        builder.Commit()
        builder.Destroy()
    # for line in lines:
    #     theSession.ActiveSketch.AddGeometry(line, NXOpen.Sketch.InferConstraintsOption.InferNoConstraints)
    # theSession.ActiveSketch.Update()

    # startPoint1 = NXOpen.Point3d(0.0, 0.0, 0.0)
    # endPoint1 = NXOpen.Point3d(4.2323860079430489, 0.0, 12.291741482791114)
    # line1 = workPart.Curves.CreateLine(startPoint1, endPoint1)
    #
    # theSession.ActiveSketch.AddGeometry(line1, NXOpen.Sketch.InferConstraintsOption.InferNoConstraints)
    #
    # theSession.ActiveSketch.Update()

    # # ----------------------------------------------
    # #   Dialog Begin Line
    # # ----------------------------------------------
    #
    # startPoint2 = NXOpen.Point3d(4.2323860079430489, 0.0, 12.291741482791114)
    # endPoint2 = NXOpen.Point3d(23.732386007943049, 0.0, 12.291741482791114)
    # line2 = workPart.Curves.CreateLine(startPoint2, endPoint2)
    #
    # theSession.ActiveSketch.AddGeometry(line2, NXOpen.Sketch.InferConstraintsOption.InferNoConstraints)
    #
    # theSession.ActiveSketch.Update()
    #
    # # ----------------------------------------------
    # #   Dialog Begin Line
    # # ----------------------------------------------
    #
    # startPoint3 = NXOpen.Point3d(23.732386007943049, 0.0, 12.291741482791114)
    # endPoint3 = NXOpen.Point3d(29.7505362394635, 0.0, 4.3053863823181615)
    # line3 = workPart.Curves.CreateLine(startPoint3, endPoint3)
    #
    # theSession.ActiveSketch.AddGeometry(line3, NXOpen.Sketch.InferConstraintsOption.InferNoConstraints)
    #
    # theSession.ActiveSketch.Update()

    # ----------------------------------------------
    #   Menu: Task->Finish Sketch
    # ----------------------------------------------
    # sketchFindMovableObjectsBuilder2 = workPart.Sketches.CreateFindMovableObjectsBuilder()
    #
    # nXObject2 = sketchFindMovableObjectsBuilder2.Commit()
    #
    # sketchFindMovableObjectsBuilder2.Destroy()
    #
    # basePart5 = theSession.Parts.BaseWork
    #
    # sketchWorkRegionBuilder1 = workPart.Sketches.CreateWorkRegionBuilder()
    #
    # sketchWorkRegionBuilder1.Scope = NXOpen.SketchWorkRegionBuilder.ScopeType.EntireSketch
    #
    # nXObject3 = sketchWorkRegionBuilder1.Commit()
    #
    # sketchWorkRegionBuilder1.Destroy()
    #
    # theSession.ActiveSketch.CalculateStatus()
    #
    # theSession.Preferences.Sketch.SectionView = False
    #
    # # markId8 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Invisible, "Deactivate Sketch")
    #
    # theSession.ActiveSketch.Deactivate(NXOpen.Sketch.ViewReorient.TrueValue, NXOpen.Sketch.UpdateLevel.Model)
    #
    # # theSession.DeleteUndoMarksSetInTaskEnvironment()
    #
    # # theSession.EndTaskEnvironment()
