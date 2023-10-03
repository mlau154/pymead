Bézier Curves
=============

The core of pymead's geometric parametrization code is built around Bézier curves. Bézier curves are a subclass
of B-splines, which are in turn a subclass of Non-Uniform Rational B-Splines (NURBS). More specifically,
Bézier curves are uniform, non-rational, clamped B-splines. **Uniform** means that Bézier curves have uniform
knot vectors. **Non-rational** means that Bézier curves can be represented by non-rational polynomials (e.g.,
:math:`t^2 + 1` as opposed to :math:`\frac{t^2+1}{t^3+2}`\ ).  **Clamped** means that Bézier curves have the useful
property that they start at their starting control point and end at their ending control point. In addition,
the local curve tangent at the endpoints is equal to slope of the line connecting the first and second or last and
second-to-last control points. This property is shown visually in the figure below.

.. figure:: images/cubic_bezier_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Cubic Bézier curve

.. figure:: images/cubic_bezier_light.*
   :width: 600px
   :align: center
   :class: only-light

   Cubic Bézier curve


Bézier curves also have the property that the control points have **global** control over the shape of the curve,
which is not generally the case with NURBS curves. **Global** control means that changing the location of a control
point changes the shape of the entire curve, except at the curve endpoints. This is illustrated in the animation below.


.. figure:: images/cubic_bezier_animated_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Cubic Bézier curve animation

.. figure:: images/cubic_bezier_animated_light.*
   :width: 600px
   :align: center
   :class: only-light

   Cubic Bézier curve animation


The degree of the Bézier curve (one less the number of control points) influences the amount of global control each
individual control point has. Increasing the number of control points further localizes the control of each individual
control point.

Mathematical Description
------------------------

Bézier curves are described by the following mathematical formula:

.. math::

   \vec{C}(t)=\sum_{i=0}^n \vec{P}_i B_{i,n}(t)

where :math:`n` is the order of the Bézier curve (equal to one less the number of control points);
:math:`\vec{P}_i` is the control point vector at the `ith` index of the form :math:`[x_i,y_i]^T`;
:math:`t` is a parameter, generally in the range :math:`[0,1]` that describes the position on the Bézier curve; and
:math:`B_{i,n}(t)` is the Bernstein polynomial, given by

.. math::

   B_{i,n}(t)={n \choose i} t^i (1-t)^{n-i}

Note that the shape of this Bernstein polynomial is influenced only by the curve degree. The curve itself
is affected by a combination of the curve degree as well as the location of the control points given by
:math:`\vec{P}_i`, which are effectively weighting values for the Bernstein polynomial.


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
