===============
Getting Started
===============

There are two main ways of using pymead: the graphical interface and the programmatic interface. For users new to
the package, the graphical interface has a shallower learning curve. The graphical interface makes it relatively
simple to design airfoils or airfoil systems with complex geometric constraints, as it allows for direct user
interaction with the airfoil graph and provides live geometry updates based on changes in the constraint equations.
The programmatic interface makes it possible to extend the functionalities of pymead to suit engineering applications
beyond the scope of the package.

Graphical Interface
===================

To open the graphical user interface (GUI) for pymead using the terminal, first navigate to the directory
containing the ``gui.py`` file (the ``gui`` module) and then run the script using Python:

.. code-block::

  cd pymead/gui
  python -m gui

Alternatively, users working in an IDE can navigate to the ``gui`` module by clicking through the directory tree.

..
    GUI documentation:

    .. raw:: html
       :file: gui_help/test.html

Programmatic Interface
======================