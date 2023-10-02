FreePoints
==========

FreePoints in pymead are the way of adding flexibility to airfoils. FreePoints are different
from AnchorPoints in that FreePoints only consist of a single Bézier control point and do
not force the Bézier curves to pass through the control point. To insert a FreePoint
from the GUI, right-click on the desired airfoil in the Parameter Tree, which brings up a
context menu as shown below:


.. figure:: images/fp_ap_menu_dark.*
   :align: center
   :class: only-dark

   Airfoil context menu


.. figure:: images/fp_ap_menu_light.*
   :align: center
   :class: only-light

   Airfoil context menu


After left-clicking on "Add FreePoint," the following dialog appears:


.. figure:: images/fp_menu_dark.*
   :align: center
   :class: only-dark

   FreePoint menu

.. figure:: images/fp_menu_light.*
   :align: center
   :class: only-light

   FreePoint menu


A description of each of these ``FreePoint`` menu items is listed below:

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


Inserting a FreePoint into the default airfoil with the default parameters shown in the
image above gives the below result. Note the additional control point located at :math:`(x,y)=(0.5,0.1)`.

.. figure:: images/free_point_insertion_dark.*
   :align: center
   :class: only-dark

   FreePoint insertion


.. figure:: images/free_point_insertion_light.*
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
