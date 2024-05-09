Adding Objects
==============

All geometric objects in `pymead` should be added to a ``GeometryCollection`` instance. A geometry collection is
found in the ``core`` module of `pymead` and can be instantiated with no arguments:

.. code-block:: python

   from pymead.core.geometry_collection import GeometryCollection

   geo_col = GeometryCollection()


Adding Points
-------------

Almost all types of objects can be added using methods of ``GeometryCollection`` that are named like
``add_<object-name>``. For example, to add a point, use the ``add_point`` method:

.. code-block:: python

   p = geo_col.add_point(0.2, -0.1)
   print(f"{p.name() = })
   print(f"{p.x().value() = })
   print(f"{p.y().value() = })


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


Adding Airfoil with Thin Trailing Edges
---------------------------------------

Airfoils with thin trailing edges are added by specifying the leading edge point and trailing edge point.

.. code-block:: python

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
   point_seq_upper = PointSequence([geo_col.add_point(xy[0], xy[1]) for xy in upper_curve_array])
   point_seq_lower = PointSequence([point_seq_upper.points()[0],
                                   *[geo_col.add_point(xy[0], xy[1]) for xy in lower_curve_array],
                                   point_seq_upper.points()[-1]])
   bez_upper = geo_col.add_bezier(point_seq_upper)
   bez_lower = geo_col.add_bezier(point_seq_lower)
   airfoil = geo_col.add_airfoil(point_seq_upper.points()[0],
                                 point_seq_upper.points()[-1],
                                 upper_surf_end=None,
                                 lower_surf_end=None
                                 )


An airfoil with a thin trailing edge can be created by simply making the upper and lower surface point sequences
end at the same point object. In that case, the ``upper_surf_end`` and ``lower_surf_end`` keyword arguments can
be omitted or set to ``None``. Note that in the above code, the same point (rather than a duplicate point)
at :math:`(0,0)` is used for the leading edge. The same is true at :math:`(1,0)` (the trailing edge).
If a duplicate point is used, a ``ClosureError`` will be raised when
trying to create the airfoil because the curves are not connected by the same point object.


Adding Airfoil with Blunt Trailing Edges
----------------------------------------

Airfoils with thin trailing edges are added by specifying the leading edge point, the trailing edge point, and the
trailing edge upper and lower surface points.
