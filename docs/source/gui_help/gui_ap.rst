AnchorPoints
============

AnchorPoints in pymead allow direct control over the airfoil surface position at a particular
location while preserving point, slope, and curvature continuity. An AnchorPoint adds five
Bézier control points to the airfoil (the minimum required to guarantee curvature continuity) and
splits the curve into curves at the middle of the five points. Because the middle control
point represents a joint between two Bézier curves, the airfoil surface is forced to pass
through this point. In the figure below, :math:`(x_0,y_0)` represents the middle control point.
The two control points to the left and the two control points to the right of this point
are also part of the AnchorPoint branch.

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

The AnchorPoint feature is powerful because it allows for inter- and intra-airfoil
positional constraints at locations other than the leading edge or trailing edge. This is
particularly useful in multi-element airfoil applications with system-level constraints,
such symmmetry, thickness at a point, or distance between two airfoil surface positions.
To insert an AnchorPoint from the GUI, right-click on the desired airfoil in the
Parameter Tree, which brings up a context menu as shown below:


.. figure:: images/fp_ap_menu_dark.*
   :align: center
   :class: only-dark

   Airfoil context menu


.. figure:: images/fp_ap_menu_light.*
   :align: center
   :class: only-light

   Airfoil context menu


After left-clicking on "Add AnchorPoint," the following dialog appears:


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
     - Downstream curvature control arm angle. This angle corresponds to the angle :math:`\psi` shown
       at the top of this page. This angle takes on the range :math:`(0, \pi)`. When :math:`\psi=0`, the control arm
       is collapsed completely inward. When :math:`\psi=\pi`, the control arm is stretched fully
       outward relative to the slope control segment. Stretching the curvature control arm past
       this angle flips the sign of the radius of curvature.
   * - psi2
     - Upstream curvature control arm angle. This angle corresponds to the angle :math:`\psi`
       closer to the leading edge than the downstream curvature control arm angle. Same range
       as psi1.
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
   * - Anchor Point Name
     - Tag for the AnchorPoint. Must not be blank and must not match any another AnchorPoint name in
       the current airfoil.


The image below shows the default airfoil with an anchor point inserted. The values corresponding
to the items in the table are shown in the Parameter Tree on the left of the image. Note that
the airfoil is now comprised of three Bézier curves (rather than two), and the Bézier curves
pass directly through the special control points marked by an "x," which correspond to the :math:`(x_0,y_0)`
point as mentioned previously.

.. figure:: images/anchor_point_insertion_dark.*
   :align: center
   :class: only-dark

   FreePoint insertion


.. figure:: images/anchor_point_insertion_light.*
   :align: center
   :class: only-light

   FreePoint insertion


..
   This HTML code adds the "only-light" and "only-dark" class to the parent figures of
   images so that the hidden figures do not take up space on the page

.. raw:: html

   <script type="text/javascript">
      var images = document.getElementsByTagName("img")
      for (let i = 0; i < images.length; i++) {
          if (images[i].classList.contains("only-light")) {
            images[i].parentNode.classList.add("only-light")
          } else if (images[i].classList.contains("only-dark")) {
            images[i].parentNode.classList.add("only-dark")
            } else {
            }
      }
   </script>
