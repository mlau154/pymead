============================================
Python Multi-Element Airfoil Design (pymead)
============================================

Welcome to the documentation page for `pymead`, an object-oriented Python 3 library and GUI for
generation, aerodynamic analysis, and aerodynamic shape optimization of parametric
airfoils and airfoil systems.


Quick Links
-----------

- `GitHub page (source code) <https://github.com/mlau154/pymead>`_
- `pymead releases page <https://github.com/mlau154/pymead/releases>`_
- `PyPi project page <https://pypi.org/project/pymead/>`_
- `pymead installation page (next tab) <https://pymead.readthedocs.io/en/latest/install.html>`_
- `GitHub issues page <https://github.com/mlau154/pymead/issues>`_


How to Use Pymead
-----------------

`pymead` has both a Graphical User Interface (GUI) and an
Application-Programming Interface (API) written in Python (see the "Architecture"
section for more details). For those less familiar with Python and those mainly seeking
to use the built-in features of `pymead`, the GUI is the main recommended method of use.
A user guide for the GUI can be found by clicking into the "GUI" tabs of the :ref:`User Guide` section.

For those desiring to extend `pymead`'s functionality, the API can be used. Even for users of the
API, an easy starting point may be to create an airfoil system with the GUI, save the
airfoil system, and load the airfoil system into the API using the ``set_from_dict_rep`` class method in
a ``GeometryCollection`` object. Basic tutorials for the API are available in the :ref:`User Guide` section,
and advanced tutorials are available in the `tutorials` subpackage in the :ref:`API` section.


Bug Squashing
-------------

If you find a bug in `pymead`, or even simply want the behavior of a feature changed or a small feature added,
feel free to create an "Issue" on the GitHub issues page (see the `Quick Links`_ section). Feature changes and
additions will be reviewed by the `pymead` developers, and bugs will be fixed in order of priority and timestamp.


Contents
--------

.. toctree::
   :maxdepth: 5

   install

.. toctree::
   :maxdepth: 2

   gallery
   user

.. toctree::
   :maxdepth: 3

   api

.. toctree::
   :maxdepth: 2

   troubleshoot
   arch
