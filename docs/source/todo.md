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
- Make the "Analysis" tab focused by default after an aerodynamics analysis (possibly implement a user option to
  override this behavior)
- Write the XFOIL/MSES analysis code using the same `CPUBoundProcess` architecture used by optimization
- Create a custom context menu for all callback plots (similar to airfoil canvas context menu) with a subset of the
  original context menu actions
- Make a "File Already Exists. Overwrite?" dialog

Refactoring
-----------
- Simplify the `PymeadObj` button implementation

Bug fixes
---------
- Fix blank line in Objective/Constraint setup not reverting to background color after editing and erasing
- Remove wave/viscous drag from XFOIL drag history plots (optimization)
- Toggle grid affects all the dock widgets
- Apply theme to status bar widgets immediately on theme change
- Correct dimensions having default colors before switching themes
- Use `QNativeGesture`/`ZoomNativeGesture` to allow pinch-zoom on macOS

Testing
-------
- Add more unit test coverage!

Documentation
-------------
- Add images to `README.md`
