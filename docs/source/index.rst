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


How to Use Pymead
-----------------

`pymead` has both a Graphical User Interface (GUI) and an
Application-Programming Interface (API) written in Python (see the "Architecture"
section for more details). For those less familiar with Python and those mainly seeking
to use the built-in features of `pymead`, the GUI is the main recommended method of use.

For those desiring to extend `pymead`'s functionality, the API can be used. Even for users of the
API, an easy starting point may be to create an airfoil system with the GUI, save the
airfoil system, and load the airfoil system into the API using the ``set_from_dict_rep`` class method in
a ``GeometryCollection`` object.

Contents
--------

.. toctree::
   :maxdepth: 5

   install

.. toctree::
   :maxdepth: 2

   gallery
   gui
   tutorials

.. toctree::
   :maxdepth: 3

   api

.. toctree::
   :maxdepth: 2

   arch
