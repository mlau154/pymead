Adding Objects
==============

All geometric objects in `pymead` should be added to a ``GeometryCollection`` instance. A geometry collection is
found in the ``core`` module of `pymead` and can be instantiated with no arguments:

.. code-block:: python

   from pymead.core.geometry_collection import GeometryCollection

   geo_col = GeometryCollection()


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
