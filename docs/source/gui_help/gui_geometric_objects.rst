Geometric Objects
#################

There are several types of geometric objects currently implemented in *pymead*
that can be useful in creating airfoil objects: points, lines, Bézier curves, polylines,
airfoils, and multi-element airfoils.

Points
======

Points are the most fundamental geometric object in *pymead*, consisting only of :math:`x`
and :math:`y` parameters. Points are used as the basis for all other types of objects.
For example, lines are always drawn between two points in *pymead*'s geometry module.

.. _point-creation:

Creation
--------

To create a point, first either press the **P** key or left-click on the "Point" button
in the toolbar (see the image below). Then, left-click on the geometry canvas to place the
point. You can continue clicking on the canvas to create additional points. Press the
**Esc** key to stop creating points.

.. |rarrow|   unicode:: U+02192 .. RIGHT ARROW

To add points from a file, select **File** |rarrow| **Import** |rarrow| **Points from File**
from the menubar. Then, click the "Choose File" button to select a text file storing a set of points. The points
should be stored row-wise, with the two columns representing the :math:`x` and :math:`y` value for each point.
The text file must have a ``.txt``, ``.dat``, or ``.csv`` file extension and be space-delimited or comma-delimited.

.. note::
   Adding points from a file is designed for convenient construction point or Bézier control point import, not
   for creating points on an airfoil surface. To import an airfoil directly from a set of coordinates in a text file,
   use the "Web Airfoil" tool (the **W** shortcut).


.. figure:: images/point_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Adding a point

.. figure:: images/point_light.*
   :width: 600px
   :align: center
   :class: only-light

   Adding a point


.. _point-modification:

Modification
------------

There are several ways to change the location of a point object:

- *Click and drag*: Hold down left-click on the point in the geometry canvas. Then, move the mouse to the desired
  location while still holding left-click.
- *Arrow keys*: To make small changes to the point's position, left-click once on the point. Then, press or hold down
  any of the arrow keys to move the point in the corresponding direction. To make larger changes, hold the **Shift**
  key while pressing/holding the arrow keys.
- *Number keys*: To directly the specify the value of the point's :math:`x` or :math:`y` position, first double-click
  on the point's name in the parameter tree (left-hand side of the figure below). Then, press the button corresponding
  either to the :math:`x` or :math:`y` value in the dialog that appears. In the final dialog, modify the value in any
  of these ways:

  - Click the up/down arrows on the right-hand side of the value spin box.
  - Click inside the value spin box and use the up/down arrows on the keyboard for small changes or the
    **Page Up**/**Page Down** keys for larger changes.
  - Select the numerical value by either triple-clicking it or by clicking inside the value spin box and pressing
    **Ctrl+A**. Then, use the number keys on the keyboard to specify a value.


.. figure:: images/point_mod_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Specifying a point's :math:`x`-value

.. figure:: images/point_mod_light.*
   :width: 600px
   :align: center
   :class: only-light

   Specifying a point's :math:`x`-value


.. _point-deletion:
Deletion
--------

To delete a single point, select the point by either left-clicking on the point in the geometry canvas or by
left-clicking on the point's name in the parameter tree. Then, delete the object by either pressing the **Delete** key
or by right-clicking on the point's name in the parameter tree and left-clicking the "Delete" option.

To delete multiple points at once, first select the points by either left-clicking on one point at a time in the
geometry canvas or by holding **Shift** or **Ctrl** and clicking the names of the points in the parameter tree. Then,
delete the points by either pressing the **Delete** key or by right-clicking on any of the selected points' names in
the parameter tree and left-clicking the "Delete" option.


.. _point-others:

Other Important Bits
--------------------
To prevent the parameter/design variable space from becoming cluttered, the :math:`x`- and :math:`y`-values of each
point do not show up under "Parameters" in the parameter tree by default. To expose the :math:`x` and :math:`y`
parameters of a particular point, right-click on the point's name in the Parameter Tree and click "Expose x and y
Parameters". For a point named "Point-1," this will add "Point-1.x" and "Point-1.y" to the "Parameters" sub-container
in the parameter tree.

To allow the optimizer to change the value of either or both of these parameters, right-click
on the newly created parameters in the parameter tree and click "Promote to Design Variable." To remove the point's
:math:`x` and :math:`y` parameters from the parameter/design variable space, right-click on the :math:`x` or :math:`y`
parameter's name in the parameter tree and click "Cover x and y Parameters."

..
   This HTML code adds the "only-light" and "only-dark" class to the parent figures of
   images so that the hidden figures do not take up space on the page


Lines
=====

Lines serve two major purposes in *pymead*: blunt trailing edge construction and flat airfoil surface section
construction. In fact, *pymead* requires that lines be used to close blunt trailing edges for Airfoil objects
to be created. These are created by default when adding an airfoil from Airfoil Tools; take the NASA supercritical
airfoil sc20012 (generated with the ``sc20012-il`` code in the "Web Airfoil" tool) for example:


.. figure:: images/te_line_sc20012_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Trailing edge line for the sc20012 airfoil

.. figure:: images/te_line_sc20012_light.*
   :width: 600px
   :align: center
   :class: only-light

   Trailing edge line for the sc20012 airfoil


Notice that two lines are created: one from the trailing edge point to the trailing edge upper surface point,
and another from the trailing edge point to the trailing edge lower surface point. More information about airfoil
trailing edges can be found in the "Airfoil" section.

Creation
--------

To create a line, first either press the **L** key or left-click on the "Line" button
in the toolbar (see the image below). Then, left-click two different points in the geometry canvas to add a line
between them. Continue to select pairs of points to add more lines, or press the **Esc** key to stop generating lines.


.. figure:: images/line_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Adding a line

.. figure:: images/line_light.*
   :width: 600px
   :align: center
   :class: only-light

   Adding a line


.. figure:: images/lines_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Adding/deleting multiple lines

.. figure:: images/lines_light.*
   :width: 600px
   :align: center
   :class: only-light

   Adding/deleting multiple lines


Deletion
--------

To delete a single line, select the line by
left-clicking on the line's name in the parameter tree. Then, delete the object by either pressing the **Delete** key
or by right-clicking on the line's name in the parameter tree and left-clicking the "Delete" option. Lines can
also be deleted by right-clicking on the line in the geometry canvas and selecting
**Modify Geometry** |rarrow| **Remove Curve** from the context menu that appears.

To delete multiple lines at once, first select the lines by holding **Shift** or **Ctrl** and clicking the names
of the lines in the parameter tree. Then,
delete the lines by either pressing the **Delete** key or by right-clicking on any of the selected lines' names in
the parameter tree and left-clicking the "Delete" option.

If either of the points associated with the line are no longer needed, the line can also be deleted by deleting either
of the points to which the line is attached (see the GIF above).


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


Creation
--------

To create a Bézier curve, first either press the **B** key or left-click on the "Bézier" button
in the toolbar (see the image below). Then, left-click at least three different points in the geometry canvas to
and press the **Enter** key to add a
line between them. Continue to select sets of three or more points to add more Bézier curves, or press the **Esc** key
to stop generating curves.

.. note::
   Creating a Bézier curve with only two control points (a *linear* Bézier curve) is effectively the same as
   generating a line! Thus, this is not allowed in *pymead* for simplicity; please use a line instead. Lines can
   be used in conjunction with Bézier curves to produce Airfoil objects.


.. figure:: images/bezier_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Adding a Bézier curve

.. figure:: images/bezier_light.*
   :width: 600px
   :align: center
   :class: only-light

   Adding a Bézier curve


Inserting & Removing Control Points
-----------------------------------

To insert a control point to an existing Bézier curve, first create a new point. Then, right-click on the curve
in the geometry canvas and select **Modify Geometry** |rarrow| **Insert Curve Point** from the context menu that
appears. Now, select first the new control point to be added, then the control point that should precede the new
control point in the control point sequence. The curve should now be updated to include the new control point, with
the curve's order increased by one. To remove a control point from the curve, simply delete the point using any of
the options in the :ref:`point-deletion` section.


.. figure:: images/add_control_point_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Adding a Bézier curve and inserting a control point

.. figure:: images/add_control_point_light.*
   :width: 600px
   :align: center
   :class: only-light

   Adding a Bézier curve and inserting a control point


Deletion
--------

To delete a single Bézier curve, select the Bézier curve by
left-clicking on the curve's name in the parameter tree. Then, delete the object by either pressing the **Delete** key
or by right-clicking on the curve's name in the parameter tree and left-clicking the "Delete" option. Curves can
also be deleted by right-clicking on the curve in the geometry canvas and selecting
**Modify Geometry** |rarrow| **Remove Curve** from the context menu that appears.

To delete multiple curves at once, first select the curves by holding **Shift** or **Ctrl** and clicking the names
of the lines in the parameter tree. Then,
delete the lines by either pressing the **Delete** key or by right-clicking on any of the selected curves' names in
the parameter tree and left-clicking the "Delete" option.

If only two or fewer of the points associated with the curve are no longer needed, the curve can also be deleted by
deleting at least all but two points to which the curve is attached.


Airfoils
========

An airfoil in *pymead* is simply defined as any closed set of curves containing a leading edge point, a trailing
edge point, and, optionally, a trailing edge upper surface point and a trailing edge lower surface point. Airfoils can
be created in several different ways in *pymead*.

Creation
--------

The primary method which gives full control over the shape of the airfoil is started using the **F** key. This method
creates an airfoil from any closed set of lines, polylines, or Bézier curves. After clicking the "Airfoil" button
or pressing the **F** key, first left-click on the leading edge point for the airfoil (either in the geometry
canvas or parameter tree). Ideally, this point is the endpoint of two of the curves in the airfoil. Then, left-click
on the trailing edge point of the airfoil.

For an airfoil with a thin trailing edge, this point must
be the endpoint of the airfoil curves. For an airfoil with a blunt trailing edge, this is not the case. In either
case, the trailing edge and leading edge points define the chord line for the airfoil, which also defines the angle of
attack. If you are designing a thin airfoil, simply press the **Enter** key after selecting both the leading edge
and trailing edge. If the points are selected properly and the airfoil is valid, the airfoil will now be shaded, and
you can hover over the shaded region to see the name of the airfoil.

For the case of a blunt trailing edge airfoil, select both the trailing edge upper surface point and trailing edge
lower surface point, in that order. Note that there must be curves connecting the trailing edge point to both of
these two points. These curves are normally lines, but they do not have to be. If the points are selected properly
and the airfoil is valid, the airfoil will now be shaded (without having to press the **Enter** key), and
you can hover over the shaded region to see the name of the airfoil.


.. figure:: images/airfoil_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Adding an airfoil with a blunt trailing edge

.. figure:: images/airfoil_light.*
   :width: 600px
   :align: center
   :class: only-light

   Adding an airfoil with a blunt trailing edge


Airfoils can also be added as polylines from coordinates.
These coordinates can originate from `Airfoil Tools <http://airfoiltools.com/>`_ or from a text/dat file.
To access either of these methods for creating an airfoil, press the "Web Airfoil" button in the toolbar or press
the **W** key. This will pop up a dialog window that looks like this:


.. figure:: images/web_airfoil_dark.*
   :width: 300px
   :align: center
   :class: only-dark

   Adding an airfoil from the web

.. figure:: images/web_airfoil_light.*
   :width: 300px
   :align: center
   :class: only-light

   Adding an airfoil from the web


Type in the tag for the airfoil (the identifier of the airfoil as shown on `Airfoil Tools <http://airfoiltools.com/>`_,
not the full name of the airfoil) in the "Web Airfoil" field. Then, press the **Enter** key. An Airfoil object
should now be present in the geometry canvas representing this airfoil. This airfoil contains only a polyline
(a series of lines connecting each subsequent pair of airfoil coordinates) and two Line objects if the airfoil
has a blunt trailing edge.

To load an airfoil from a ``.txt``, ``.dat``, or ``.csv`` file, press the "Web Airfoil" button in the toolbar or press
the **W** key. Now, select the "Coordinate File" option from the drop-down menu. Then, press the "Select Airfoil"
button to select a file. The file has to have one of the aforementioned extensions, and the coordinates should
be listed row-wise, starting at the trailing edge upper surface point and moving counter-clockwise to the trailing
edge lower surface point. The file must also be space-delimited.


.. figure:: images/airfoil_from_file_dark.*
   :width: 300px
   :align: center
   :class: only-dark

   Adding an airfoil from a text file

.. figure:: images/airfoil_from_file_light.*
   :width: 300px
   :align: center
   :class: only-light

   Adding an airfoil from a text file


Deletion
--------

An airfoil can be deleted by left-clicking the airfoil's name in the parameter tree and pressing the **Delete** key or
by right-clicking the airfoil's name in the parameter tree and clicking the **Delete** option from the context menu
that appears. Alternatively, an airfoil can be deleted by deleting any of its associated points or curves.


Multi-Element Airfoils
======================

Multi-element airfoils do not have any inherent geometric representation, but are simply unordered collections
of Airfoil objects. These multi-element airfoils must be created, even in the case of a single airfoil, to run
an MSES analysis or optimization.

Creation
--------

To create a multi-element airfoil, select the "Multi-Element Airfoil" button from the toolbar or press the **M** key.
Then, hold **Shift** or **Ctrl** while left-clicking each of the airfoil names from the parameter tree. Then,
click anywhere on the geometry canvas and press the **Enter** key.


.. figure:: images/mea_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Adding a multi-element airfoil

.. figure:: images/mea_light.*
   :width: 600px
   :align: center
   :class: only-light

   Adding a multi-element airfoil


Deletion
--------

A multi-element airfoil can be deleted by left-clicking the multi-element airfoil's name in the parameter tree and
pressing the **Delete** key or
by right-clicking the multi-element airfoil's name in the parameter tree and clicking the **Delete** option
from the context menu that appears.


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
