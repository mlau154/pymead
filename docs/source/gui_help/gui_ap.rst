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
