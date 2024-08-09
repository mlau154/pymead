Renaming Objects
################

.. tab-set::

    .. tab-item:: GUI
        :sync: gui

        *Construction Zone*

    .. tab-item:: API
        :sync: api

        Objects can be assigned custom names at or after instantiation. To assign a name at instantiation, simply assign
        a ``str`` to the ``name`` keyword argument in any of the ``add_<object-name>`` methods of a ``GeometryCollection``.
        For example, to name the point :math:`(1,0)` as a trailing edge, use

        .. code-block:: python

           from pymead.core.geometry_collection import GeometryCollection

           geo_col = GeometryCollection()
           geo_col.add_point(1.0, 0.0, name="TrailingEdge")


        Or, equivalently,

        .. code-block:: python

           from pymead.core.geometry_collection import GeometryCollection

           p1 = geo_col.add_point(1.0, 0.0)
           p1.set_name("TrailingEdge")
