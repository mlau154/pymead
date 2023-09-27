Airfoil Generation
==================

The following steps show how to create and save an airfoil from the pymead GUI.

Next, we will insert some ``FreePoint``\ s into the airfoil, which adds additional degrees
of freedom to individual Bézier curves. To insert a ``FreePoint`` from the GUI, right-click
on the desired airfoil in the Parameter Tree, which brings up a context menu as shown below:


.. figure:: images/fp_ap_menu_dark.*
   :align: center
   :class: only-dark

   Airfoil context menu

.. figure:: images/fp_ap_menu_light.*
   :align: center
   :class: only-light

   Airfoil context menu


This brings up a dialog as shown below.


.. figure:: images/fp_menu_dark.*
   :align: center
   :class: only-dark

   FreePoint menu

.. figure:: images/fp_menu_light.*
   :align: center
   :class: only-light

   FreePoint menu


Here is a description of each of the ``FreePoint`` menu items:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Item
     - Parameter
   * - x
     - Distance from the origin in the "Geometry" window along the x-axis
   * - y
     - Distance from the origin in the "Geometry" window along the y-axis
   * - Previous Anchor Point
     - Parent ``AnchorPoint`` to which this ``FreePoint`` belongs. More specifically, the ``FreePoint``
       will be inserted into the Bézier curve which has this ``AnchorPoint`` as its first ``ControlPoint``
       using counter-clockwise ordering. For an airfoil with no custom ``AnchorPoint``\ s, inserting this
       ``FreePoint`` with the Previous Anchor Point set to ``"te_1"`` corresponds to adding a control
       point to the airfoil's upper surface, while inserting a ``FreePoint`` with the
       Previous Anchor Point set to ``"le"`` corresponds to adding a control point to the airfoil's
       lower surface. Note that ``"te_1"`` represents the upper trailing edge point,
       which is distinct from the lower trailing edge point in the case of an airfoil with a blunt
       trailing edge.
   * - Previous Free Point
     - Similar to the "Previous Anchor Point" item, this item sets the ``FreePoint`` insertion index
       within the Bézier curve's control point matrix using counter-clockwise ordering. The
       difference here is that an existing ``FreePoint`` is specified, rather than an ``AnchorPoint``.
       Note that if no ``FreePoint``\ s have been added yet to the Bézier curve corresponding to the
       ``AnchorPoint`` specified by "Previous Anchor Point", ``None`` is automatically selected.

.. figure:: images/ap_menu_dark.*
   :align: center
   :class: only-dark

   AnchorPoint menu

.. figure:: images/ap_menu_light.*
   :align: center
   :class: only-light

   AnchorPoint menu

Here is a description of each of the ``AnchorPoint`` menu items:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Item
     - Parameter
   * - x
     - Distance from the origin in the "Geometry" window along the x-axis
   * - y
     - Distance from the origin in the "Geometry" window along the y-axis
   * - L
     - Distance between the control points located immediately before and after the ``AnchorPoint``
       in the counter-clockwise ordering divided by the chord length of the airfoil.
   * - R
     - Radius of curvature of the airfoil at the ``AnchorPoint`` divided by the chord length of the
       airfoil.
   * - r
     - Ratio of the distance between the control point immediately upstream of the ``AnchorPoint`` and
       the ``AnchorPoint`` itself to the distance between the control points located immediately before
       and after the ``AnchorPoint``\ . Here, "upstream" means after the ``AnchorPoint`` in the
       counter-clockwise ordering for an ``AnchorPoint`` on the upper surface and before the
       ``AnchorPoint`` for an ``AnchorPoint`` on the lower surface. This is normally in the
       range :math:`[0,1]`; however, this is not enforced.
   * - phi
     - "Tilt" of the line connecting the control points immediately before and after the ``AnchorPoint``.
       Regardless of whether the ``AnchorPoint`` is located on the airfoil upper surface or lower surface,
       positive values of "phi" tilt the line toward the leading edge, while negative values of "phi"
       tilt the line away from the leading edge.
   * - psi1
     -
   * - Previous Anchor Point
     - Parent ``AnchorPoint`` to which this ``FreePoint`` belongs. More specifically, the ``FreePoint``
       will be inserted into the Bézier curve which has this ``AnchorPoint`` as its first ``ControlPoint``
       using counter-clockwise ordering. For an airfoil with no custom ``AnchorPoint``\ s, inserting this
       ``FreePoint`` with the Previous Anchor Point set to ``"te_1"`` corresponds to adding a control
       point to the airfoil's upper surface, while inserting a ``FreePoint`` with the
       Previous Anchor Point set to ``"le"`` corresponds to adding a control point to the airfoil's
       lower surface. Note that ``"te_1"`` represents the upper trailing edge point,
       which is distinct from the lower trailing edge point in the case of an airfoil with a blunt
       trailing edge.
   * - Previous Free Point
     - Similar to the "Previous Anchor Point" item, this item sets the ``FreePoint`` insertion index
       within the Bézier curve's control point matrix using counter-clockwise ordering. The
       difference here is that an existing ``FreePoint`` is specified, rather than an ``AnchorPoint``.
       Note that if no ``FreePoint``\ s have been added yet to the Bézier curve corresponding to the
       ``AnchorPoint`` specified by "Previous Anchor Point", ``None`` is automatically selected.


.. figure:: images/bezier_g2_diagram_dark.*
   :width: 500px
   :align: center
   :class: only-dark

   Bézier G\ :sup:`2` continuity

.. figure:: images/bezier_g2_diagram_light.*
   :width: 500px
   :align: center
   :class: only-light

   Bézier G\ :sup:`2` continuity
