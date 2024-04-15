Geometric Objects
#################

There are several types of geometric objects currently implemented in *pymead*
that can be useful in creating airfoil objects: points, lines, Bézier curves,
airfoils, and multi-element airfoils.

Points
======

Points are the most fundamental geometric object in *pymead*, consisting only of :math:`x`
and :math:`y` parameters. Points are used as the basis for all other types of objects.
For example, lines are always drawn between two points in *pymead*'s geometry module.

.. _point-creation:

Creation
--------

To create a point, first either press the **P** key or left-click on the "point" button
in the toolbar (see the image below). Then, left-click on the geometry canvas to place the
point. You can continue clicking on the canvas to create additional points. Press the
**Esc** key to stop creating points.


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
