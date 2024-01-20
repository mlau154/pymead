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
- Add the capability to remove constraints
- Add absolute angle constraints
- Add variable constraint GUI object positioning (add keyword arguments that are stored in the `.jmea` files)
- Add error handling for over-constrained/Jax shape/no solution in GCS
- Add an undo/redo framework to the GUI

Refactoring
-----------
- Unify the implementation of API-GUI canvas elements
- Simplify the `PymeadObj` button implementation

Bug fixes
---------
- May need to eliminate the use of modulo in constraint equations to make the variable 
  space smooth
