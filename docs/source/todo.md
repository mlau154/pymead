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
  original context menu action
- Use `QNativeGesture`/`ZoomNativeGesture` to allow pinch-zoom on macOS
- Add modifiable perpendicular constraint handle positioning
- Add handle offsets to `.jmea` files
- Add the ability to hide individual objects
- Direct & inverse airfoil design modules
- Add polyline from GUI
- Add downsampling feature to XFOIL
- Make downsampling also apply to polylines
- Make single-step unit aware for spin-boxes and arrow-key movements
- Add XFOIL polar feature
- Make web airfoils transformable (dialog and canvas-dynamic?) and create design variables based on
  c, alf, dx, dy
- Add XFOIL flap deflection feature
- Use `.jses` extension for MSES settings and `.jopt` for optimization settings files
- Make geometric classes available from the top-level module
- Add downsampling option to make number of sampled points on each curve proportional to arc length
- Show "Untitled" instead of nothing for the current save name if a new file was created and not
  yet saved
- Add asterisk when an undo_redo action is triggered and remove when save is called
- Add save protection for example files
- Add a "view data" feature to .jmea, .jses, and .jopt files that allows users to view the JSON
  data inside a custom text browser
- Autofocus angle of attack value when loading Ctrl+Alt+I inviscid analysis
- Make a "jump to tree location" option in context menu when any `PymeadObj` is selected
- Add a unit selector to the Export IGES dialog
- More verbose reasons for why MSES is failing (diverged, timeout, etc.)

Refactoring
-----------
- Simplify the `PymeadObj` button implementation
- Delete obsolete dialogs, dialog widgets, and dialog settings json files

Bug fixes
---------
- Base angle dimension change in settings menu not working properly
- Fix blank line in Objective/Constraint setup not reverting to background color after editing and erasing
- Remove wave/viscous drag from XFOIL drag history plots (optimization)
- Fix bug where when trying to create a new optimization directory using the root directory but a directory was renamed
  after a previous optimization to the same name but with additional numbers and strings following the root name
- Fix draggable constraint handles interfering with each other
- Fix angle constraints sometimes only responding to movement in the antiparallel direction
- Fix visual artifacts of FramelessWindow appearing while dragging
- Fix some airfoils of an MEA occasionally not being added to MSET dialog
- Fix XFOIL/MSES/Opt settings not updating properly (including loading MSES settings in optimization setup overriding 
  the dialog window title)
- Add last-resort forceful process termination during closeEvent (especially from terminal `pymead-gui` command)
- Fix bug where updating constraint value by typing text in the canvas leaves the visual value out of sync
  when the constraint parameter is a design variable and the requested value is out of bounds
- Point.x and Point.y desvar/params possibly do not get deleted properly after their parent point is deleted
- MPOLAR key error when loading MSES settings into optimization settings (maybe make MPOLAR a feature here?)
- Store algorithm data as JSON to fix package version error when loading in `.pkl` files from
  a different environment than they were saved on
- Fix bug where loading in an airfoil coordinate file with a header changes the result instead of erroring out
- Unselecting "Actuator Disks Active" after adding an actuator disk gives attribute error:
  "'PymeadLabeledPushButton' object has no attribute 'setReadOnly'"

Testing
-------
- Add more unit test coverage!
