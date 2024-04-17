Overview
########

The graphical user interface in *pymead* has several components, each of which has a unique role.

.. figure:: images/gui_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Elements of the *pymead* GUI

.. figure:: images/gui_light.*
   :width: 600px
   :align: center
   :class: only-light

   Elements of the *pymead* GUI


.. |rarrow|   unicode:: U+02192 .. RIGHT ARROW


- The **Title Bar** shows the name of the currently loaded ``.jmea`` file (the `JSON <https://www.json.org/>`_-based
  storage formats for geometry collections in *pymead*.
- The **Menu Bar** houses most of the actions available in *pymead*, including file loading/importing/saving,
  aerodynamic analysis tools, and optimization commands.
- The **Toolbar** contains some of the most frequently used actions in *pymead*, including the buttons for
  geometry object creation.
- The **Parameter Tree** houses a nested list of all the geometric objects presently loaded, regardless of whether
  they are visible in the geometry canvas.
- The **Geometry Canvas** contains a visible representation of all the geometric objects, including, points, lines,
  curves, and airfoils. The canvas is interactive and allows for the typical pan/zoom actions
  (use **View** |rarrow| **Fit** to see everything on the screen at once).
- Directly underneath the geometry canvas are the tabs that can be used to access the **Analysis Windows**.
  These windows appear when aerodynamic analyses or shape optimization studies are run using XFOIL or MSES
  (see the **Analysis** and **Optimization** items in the menu bar).
- The **Console** contains important output information from analyses and optimization runs. All text from the console
  can be selected with **Ctrl+A** and copied with **Ctrl+C**.
- The **Status Bar** shows various tool tips, as well as live information during optimization runs. The status bar
  also contains a drop-down menu for inviscid lift coefficient analysis of airfoils that can be selected when there
  are one or more airfoils present in the geometry collection. Finally, the version of *pymead* in use is shown
  in the bottom right-hand corner.

The *pymead* GUI is fully customizable; the parameter tree, console, geometry canvas, and analysis windows can all
be arranged according to preference, or even undocked to form separate windows. To re-arrange the windows,
left-click and drag the top of the widget, near where the title of the widget is shown (e.g., "Tree"). To undock,
drag the widget outside of the GUI. To redock, drag the widget back inside the GUI. The widgets can also be
stretched or shrunk by left-clicking and dragging the widget dividers (the small sets of six dots between the widgets).


.. figure:: images/gui_customize_dark.*
   :width: 600px
   :align: center
   :class: only-dark

   Customization of the *pymead* GUI

.. figure:: images/gui_customize_light.*
   :width: 600px
   :align: center
   :class: only-light

   Customization of the *pymead* GUI


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