Airfoil Matching
################

A common problem in airfoil design is matching an existing set of airfoil coordinates with a parametrization. For
example, a design strategy might call for optimizing an airfoil using a NACA 0012 as a baseline airfoil. However,
the NACA 0012 is defined by a simple polynomial function that might not be a useful parametrization scheme for more
complex airfoil designs. Therefore, we can take an existing parametrization and find the set of design variables that
makes the output airfoil most closely match the target NACA 0012 airfoil.

To start, we first add an ``Airfoil`` object to the canvas. See :ref:`geo-objs` for a guide on how to do this. Then,
we need an airfoil coordinate file representing the target airfoil. This can be either the tag of an airfoil on
`Airfoil Tools <http://airfoiltools.com>`_ or an airfoil coordinate file. The coordinate file must be in the `Selig`
format, where the first point is the upper surface of the trailing edge, and the points move counter-clockwise to
the lower surface of the trailing edge.

.. tab-set::

    .. tab-item:: GUI
        :sync: gui

        *Construction Zone*

    .. tab-item:: API
        :sync: api

        *Construction Zone*

