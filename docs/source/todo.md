Developer To-Dos
================

Installation upgrades
---------------------
- Add `apt` package in Linux

Planned feature additions
-------------------------
- Add the ability to flip/switch constraint reference points
- Add variable constraint GUI object positioning (add keyword arguments that are stored in the `.jmea` files)
- Graphical highlighting of constraint parameters and constraints - make hoverEnter detection on constraint canvas items
- Tie parameter hover to associated constraint hover events
- Write the XFOIL analysis code using the same `CPUBoundProcess` architecture used by optimization and MSES
- Create a custom context menu for all callback plots (similar to airfoil canvas context menu) with a subset of the
  original context menu actions
- Make a "File Already Exists. Overwrite?" dialog

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

Testing
-------
- Add more unit test coverage!
