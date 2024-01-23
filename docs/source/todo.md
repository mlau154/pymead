Developer To-Dos
================

Installation upgrades
---------------------
- Add pip install option
- Add `apt` package in Linux
- Add macOS support?

Planned feature additions
-------------------------
- Add the ability to fix points
- Add strong absolute angle constraints
- Add the ability to flip/switch constraint reference points
- Add variable constraint GUI object positioning (add keyword arguments that are stored in the `.jmea` files)
- Add error handling for over-constrained/Jax shape/no solution in GCS
- Add an undo/redo framework to the GUI
- Graphical highlighting of constraint parameters and constraints
- Add unit selection ComboBox
- Add renaming to parameters
- Add metadata to save files (date, pymead version, etc.)
- Make the "Analysis" tab focused by default after an aerodynamics analysis (possibly implement a user option to
  override this behavior)
- Write the XFOIL/MSES analysis code using the same `CPUBoundProcess` architecture used by optimization
- Re-implement downsampling (method stored in Airfoil --> MEA --> Optimization)
- Create a custom context menu for all callback plots (similar to airfoil canvas context menu) with a subset of the
  original context menu actions

Refactoring
-----------
- Unify the implementation of API-GUI canvas elements
- Simplify the `PymeadObj` button implementation
- Remove print messages!

Bug fixes
---------
- May need to eliminate the use of modulo in constraint equations to make the variable 
  space smooth
- Fix loading bar for new version
- Fix blank line in Objective/Constraint setup not reverting to background color after editing and erasing
- Remove wave/viscous drag from XFOIL drag history plots (optimization)
- Fix symmetry constraint having switched target/tool points (perhaps automatically create the mirror point?)
- Airfoil comboboxes not properly updating when an optimization settings file is loaded without a GeometryCollection
  present
- Toggle grid affects all the dock widgets

Testing
-------
- Add more unit test coverage!
