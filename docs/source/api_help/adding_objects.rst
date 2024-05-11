Adding Objects
==============

All geometric objects in `pymead` should be added to a ``GeometryCollection`` instance. A geometry collection is
found in the ``core`` module of `pymead` and can be instantiated with no arguments:

.. code-block:: python

   from pymead.core.geometry_collection import GeometryCollection

   geo_col = GeometryCollection()


It may be desirable in some circumstances to load in a geometry collection saved from the GUI (``.jmea`` files).
In this case,
first load the geometry collection into a Python dictionary, then pass that dictionary to the static method
``GeometryCollection.set_from_dict_rep``:

.. code-block:: python

   from pymead.utils.read_write_files import load_data

   my_pymead_data = load_data("my_jmea_file.jmea")
   geo_col = GeometryCollection.set_from_dict_rep(my_pymead_data)


This method is also how saved ``.jmea`` files are loaded in the GUI.


Adding Points
-------------

Almost all types of objects can be added using methods of ``GeometryCollection`` that are named like
``add_<object-name>``. For example, to add a point, use the ``add_point`` method:

.. code-block:: python

   p = geo_col.add_point(0.2, -0.1)
   print(f"{p.name() = }")
   print(f"{p.x().value() = }")
   print(f"{p.y().value() = }")


Notice that these object-add methods always return the object that is created. The above code block illustrates how to
access attributes of the object. If the object instance is assigned to a variable (``p`` in the previous example),
the object can be accessed from the geometry collection's ``container`` dictionary. For example, to access the point
object, the following code can be used after the point is added:

.. code-block:: python

   p = geo_col.container()["points"]["Point-1"]


Each object type is sorted into a sub-container (``"points"`` in the previous example). The names of each of these
sub-containers is given as follows (these names can also be shown by printing ``geo_col.container().keys()``):

- Design variables: ``"desvar"``
- Parameters: ``"params"``
- Points: ``"points"``
- Lines: ``"lines"``
- Polylines: ``"polylines"``
- Bézier curves: ``"bezier"``
- Airfoils: ``"airfoils"``
- Multi-element airfoils: ``"mea"``
- Geometric constraints: ``"geocon"``


Point Sequences
---------------

The ``PointSequence`` object is used to create instances of lines and Bézier curves. This type of object
adds several convenience methods for adding, reversing, inserting, etc. Point sequences can be generated using
previously created points in the geometry collection, or more directly from a ``numpy`` array.

.. code-block:: python

   import numpy as np
   from pymead.core.point import PointSequence

   p1 = geo_col.add_point(0.3, 0.2)
   p2 = geo_col.add_point(0.4, -0.5)
   p3 = geo_col.add_point(0.7, 0.7)
   point_seq_1 = PointSequence([p1, p2, p3])

   point_array = np.array([[0.3, 0.2], [0.4, -0.5], [0.7, 0.7]])
   point_seq_2 = PointSequence.generate_from_array(point_array)


Adding Lines & Bézier Curves
----------------------------

Line segments and Bézier curves are both added by passing a ``PointSequence`` object to the constructor.

.. code-block:: python

   p1 = geo_col.add_point(0.3, 0.2)
   p2 = geo_col.add_point(0.4, -0.5)
   p3 = geo_col.add_point(0.7, 0.7)
   line = geo_col.add_line(PointSequence([p1, p2]))
   bez = geo_col.add_bezier(PointSequence([p1, p2, p3]))


Adding Airfoils with Thin Trailing Edges
----------------------------------------

Airfoils with thin trailing edges are added by specifying the leading edge point and trailing edge point.

.. code-block:: python

   # Define the array of control points for the airfoil's Bézier curves
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

   # Generate the point sequences
   point_seq_upper = PointSequence([geo_col.add_point(xy[0], xy[1]) for xy in upper_curve_array])
   point_seq_lower = PointSequence([point_seq_upper.points()[0],
                                   *[geo_col.add_point(xy[0], xy[1]) for xy in lower_curve_array],
                                   point_seq_upper.points()[-1]])

   # Add the Bézier curves
   bez_upper = geo_col.add_bezier(point_seq_upper)
   bez_lower = geo_col.add_bezier(point_seq_lower)

   # Create the airfoil
   airfoil = geo_col.add_airfoil(point_seq_upper.points()[0],
                                 point_seq_upper.points()[-1],
                                 upper_surf_end=None,
                                 lower_surf_end=None
                                 )


Note that in the above code, the same point (rather than a duplicate point)
at :math:`(0,0)` is used for the leading edge. The same is true at :math:`(1,0)` (the trailing edge).
If a duplicate point is used, a ``ClosureError`` will be raised when
trying to create the airfoil because the curves are not connected by the same point object. To visualize the airfoil,
simply do

.. code-block:: python

   airfoil.plot()


See the API documentation for details on the possible keyword arguments for this method.


Adding Airfoils with Blunt Trailing Edges
-----------------------------------------

Airfoils with blunt trailing edges are added by specifying the leading edge point, the trailing edge point, and the
trailing edge upper and lower surface points. Note that for airfoils with blunt trailing edges, lines or curves
connecting the upper surface point to the trailing edge and lower surface point to the trailing edge must be present.

.. code-block:: python

   # Define the array of control points for the airfoil's Bézier curves
   upper_curve_array = np.array([
       [0.0, 0.0],
       [0.0, 0.05],
       [0.05, 0.05],
       [0.6, 0.04],
       [1.0, 0.0025]
   ])
   lower_curve_array = np.array([
       [0.0, -0.05],
       [0.05, -0.05],
       [0.7, 0.01],
       [1.0, -0.0025]
   ])

   # Generate the point sequences
   point_seq_upper = PointSequence([geo_col.add_point(xy[0], xy[1]) for xy in upper_curve_array])
   point_seq_lower = PointSequence([point_seq_upper.points()[0],
                                   *[geo_col.add_point(xy[0], xy[1]) for xy in lower_curve_array]])

   # Add the Bézier curves
   bez_upper = geo_col.add_bezier(point_seq_upper)
   bez_lower = geo_col.add_bezier(point_seq_lower)

   # Add the trailing edge at (1, 0)
   te_point = geo_col.add_point(1.0, 0.0)

   # Add lines connecting the trailing edge to the trailing edge upper and lower points
   te_upper_line = geo_col.add_line(PointSequence([point_seq_upper.points()[-1], te_point]))
   te_lower_line = geo_col.add_line(PointSequence([point_seq_lower.points()[-1], te_point]))

   # Create the airfoil
   airfoil = geo_col.add_airfoil(leading_edge=point_seq_upper.points()[0],
                                 trailing_edge=te_point,
                                 upper_surf_end=point_seq_upper.points()[-1],
                                 lower_surf_end=point_seq_lower.points()[-1]
                                 )


Adding Multi-Element Airfoils
-----------------------------

For performing analysis or optimization using MSES, even if a single airfoil is being studied, it
is necessary to place the airfoil(s) in a multi-element airfoil container. As an example, this can be done for three
airfoil objects named ``"airfoil_1"``, ``"airfoil_2"``, and ``"airfoil-3"``, each created in a similar fashion to
those in the above code:

.. code-block:: python

   mea = geo_col.add_mea([airfoil_1, airfoil_2, airfoil_3])
