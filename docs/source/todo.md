Developer To-Dos
================

Installation upgrades
---------------------
- Add `apt` package in Linux

Planned feature additions
-------------------------
- Add the ability to flip/switch constraint reference points
- Graphical highlighting of constraint parameters and constraints - make hoverEnter detection on constraint canvas items
- Tie parameter hover to associated constraint hover events
- Write the XFOIL analysis code using the same `CPUBoundProcess` architecture used by optimization and MSES
- Create a custom context menu for all callback plots (similar to airfoil canvas context menu) with a subset of the
  original context menu actions
- Make a "File Already Exists. Overwrite?" dialog
- Add save protection for example files
- Add modifiable perpendicular constraint handle positioning
- Add handle offsets to `.jmea` files
- Add the ability to hide individual objects
- Display RAM/CPU usage live during optimization (using `psutil.virtual_memory().percent` and `psutil.cpu_percent()`)
- Direct & inverse airfoil design modules
- Add polyline from GUI
- Make function tolerance default value dependent on number of objectives
- Add downsampling feature to XFOIL
- Make downsampling also apply to polylines
- Add variable number of Bezier evaluation points in the `BezierButton` dialog
- Make single-step unit aware for spin-boxes and arrow-key movements
- Make airfoil-relative points an option, either by adding this as a new constraint option
  or by adding a list of airfoil-relative points to that particular airfoil. Either way,
  the points would need to be added to the graph
- Add XFOIL polar feature
- Make web airfoils transformable (dialog and canvas-dynamic?) and create design variables based on
  c, alf, dx, dy
- Add XFOIL flap deflection feature

Refactoring
-----------
- Simplify the `PymeadObj` button implementation

Bug fixes
---------
- Base angle dimension change in settings menu not working properly
- Fix blank line in Objective/Constraint setup not reverting to background color after editing and erasing
- Remove wave/viscous drag from XFOIL drag history plots (optimization)
- Use `QNativeGesture`/`ZoomNativeGesture` to allow pinch-zoom on macOS
- Fix bug where when trying to create a new optimization directory using the root directory but a directory was renamed
  after a previous optimization to the same name but with additional numbers and strings following the root name
- Fix draggable constraint handles interfering with each other
- Fix angle constraints sometimes only responding to movement in the antiparallel direction
- Create GUI error message instead of early terminating for plotting Mach contours etc. with MSES analysis if MuPDF
  or ps2pdf not found
- Fix visual artifacts of FramelessWindow appearing while dragging
- Throw GUI error if XFOIL airfoil has more than 495 coordinate points (hard-coded limit)
- Fix the following `QLayout` warning when loading downsampling preview:
  'Attempting to add QLayout "" to DownsamplingPreviewDialog "", which already has a layout'
- Fix some airfoils of an MEA occasionally not being added to MSET dialog
- Fix no GUI error being thrown when trying to visualize downsampling on an empty/non-existent MEA
- Fix bug where multiprocessing pool does not properly terminate

Aesthetics
----------
- Make airfoil coordinate downsampling preview adhere to the global format

Testing
-------
- Add more unit test coverage!
