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


def add_sketch(points: list, curve_orders: list):
    theSession = NXOpen.Session.GetSession()
    workPart = theSession.Parts.Work
    displayPart = theSession.Parts.Display
    # ----------------------------------------------
    #   Menu: Insert->Sketch
    # ----------------------------------------------
    # markId1 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Enter Sketch")
    #
    # markId2 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Update Model from Sketch")
    #
    # theSession.BeginTaskEnvironment()
    #
    # markId3 = theSession.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "Start")

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

    theSession.Preferences.Sketch.ScaleOnFirstDrivingDimension = True

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

    point_list = [NXOpen.Point3d(point[0], point[1], point[2]) for point in points]
    lines = [workPart.Curves.CreateLine(point_list[idx], point_list[idx + 1]) for idx in range(len(point_list) - 1)]

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

    def generate_splines():
        starting_point = 0
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

    generate_splines()

    # # ================================= ADD ANGULAR DIMENSION =======================================

    sketchAngularDimensionBuilder2 = workPart.Sketches.CreateAngularDimensionBuilder(
        NXOpen.Annotations.AngularDimension.Null)

    sketchAngularDimensionBuilder2.Driving.DrivingMethod = NXOpen.Annotations.DrivingValueBuilder.DrivingValueMethod.Driving

    # sketchAngularDimensionBuilder2.Driving.DimensionValue = 139.0

    scalar1 = workPart.Scalars.CreateScalar(0.0, NXOpen.Scalar.DimensionalityType.NotSet,
                                            NXOpen.SmartObject.UpdateOption.WithinModeling)
    point1 = workPart.Points.CreatePoint(lines[2], scalar1, NXOpen.SmartObject.UpdateOption.WithinModeling)
    point1_3d = point1.Coordinates
    # point1_3d = NXOpen.Point3d(0.0, 0.0, 0.0)

    sketchAngularDimensionBuilder2.FirstAssociativity.SetValue(lines[2], workPart.ModelingViews.WorkView, point1_3d)

    scalar2 = workPart.Scalars.CreateScalar(1.0, NXOpen.Scalar.DimensionalityType.NotSet,
                                            NXOpen.SmartObject.UpdateOption.WithinModeling)
    point2 = workPart.Points.CreatePoint(lines[3], scalar2, NXOpen.SmartObject.UpdateOption.WithinModeling)

    point2_3d = point2.Coordinates
    # point2_3d = NXOpen.Point3d(0.5, 0.0, 0.0)

    sketchAngularDimensionBuilder2.SecondAssociativity.SetValue(lines[3], workPart.ModelingViews.WorkView, point2_3d)

    sketchAngularDimensionBuilder2.Commit()

    sketchAngularDimensionBuilder2.Destroy()

    # =============================== END ADD ANGULAR DIMENSION =====================================

    theSession.ActiveSketch.Update()

    theSession.ActiveSketch.Deactivate(NXOpen.Sketch.ViewReorient.TrueValue, NXOpen.Sketch.UpdateLevel.Model)

    # theSession.DeleteUndoMarksSetInTaskEnvironment()
    #
    # theSession.EndTaskEnvironment()


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
