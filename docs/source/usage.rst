=======
Install
=======

Installation methods
====================

There are several easy ways to install pymead:

Method 1: ``pip``
-----------------
Use ``pip`` to install the latest stable version of pymead into the environment from the Python Package Index (PyPi):

.. code-block::

  pip install pymead

This method automatically installs all required dependencies that are not yet installed. It also
allows the user to easily update pymead if desired when a new version is available.

Method 2: IDE
-------------
Some IDEs, like `PyCharm <https://www.jetbrains.com/pycharm/>`_, have a plugin for ``pip``. In PyCharm,
simply search for and install "pymead" in the "Python Packages" tab.

Method 3: Local install
-----------------------
The pymead package can also be installed in a local location. To accomplish this, first clone the directory using

.. code-block::

  git clone https://github.com/mlau154/pymead.git

If installing the most recent version of pymead, the branch corresponding to that version must be checked out.
For instance, to checkout version 2.0.0, use

.. code-block::

  git checkout v2.0.0

To see all available branches, use

.. code-block::

  git branch -a

To pull the latest changes from the repository at some point after installation, use

.. code-block::

  git pull

Cloning and checking out a branch of the pymead repository only copies the source code into a directory. Installation
after this step is still recommended because installation automatically installs all Python dependencies and makes the
pymead package importable from outside the repository. To install, navigate to the top-level directory of the install
location in the terminal (where the ``setup.py`` file is located), then type:

.. code-block::

  pip install .

This will install pymead and all of its dependencies into the ``Lib/site-packages`` folder of the current version of
Python. To check that the installation succeeded, run the following lines of code in your Python interpreter:

.. code-block:: python

  from pymead.core.airfoil import Airfoil
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  a = Airfoil()
  a.plot_airfoil(ax)
  ax.set_aspect("equal")
  plt.show()

If the installation was successful, an airfoil will appear on the screen.

Dependencies
============

Required
--------

Each of the following dependencies are required to use pymead. All packages listed in this section are automatically
installed when using ``pip install pymead``. Alternatively, if cloning from the
`pymead GitHub repo <https://github.com/mlau154/pymead>`_, these requirements can be installed using the included
``requirements.txt`` file.

- `scipy <https://scipy.org/>`_: Used for airfoil matching
- `numpy <https://numpy.org/>`_: Used for linear algebra and random numbers
- `shapely <https://shapely.readthedocs.io/en/stable/>`_: Computational geometry
- `matplotlib <https://matplotlib.org/>`_: Static plotting
- `requests <https://requests.readthedocs.io/en/latest/>`_: Downloading airfoil coordinate sets
  from `Airfoil Tools <http://airfoiltools.com/>`_
- `PyQt5 <https://pypi.org/project/PyQt5/>`_: Graphical User Interface (GUI)
- `pyqtgraph <https://www.pyqtgraph.org/>`_: Interactive plotting and parameter trees
- `python-benedict <https://pypi.org/project/python-benedict/>`_: Dictionary utilities
- `pandas <https://pandas.pydata.org/>`_: Data structures
- `pymoo <https://pymoo.org/>`_: Aerodynamic shape optimization
- `numba <https://numba.pydata.org/>`_: Speed-up of inviscid lift coefficient calculation

Optional
--------

pymead relies on several external libraries for low- and medium-fidelity
aerodynamic analysis. All the geometry tools in pymead are built-in, apart
from several Python libraries that are installed automatically if ``pip`` is used
as the install method. However, parts of the ``analysis`` and ``optimization`` modules
cannot be used without the separate installation of the following external libraries. To
make the most of pymead, download the following software packages:

- `XFOIL <https://web.mit.edu/drela/Public/web/xfoil/>`_: low-fidelity,
  single-airfoil-element aerodynamic analysis (linear-strength vortex
  panel code coupled with a boundary-layer model)
- `MSES <https://tlo.mit.edu/technologies/mses-software-high-lift-multielement-airfoil-configurations>`_:
  medium-fidelity, multi-airfoil-element aerodynamic analysis (Euler-equation
  solver coupled with the same boundary-layer model as XFOIL)
- `Ghostscript <https://www.ghostscript.com/>`_: PS-to-PDF file conversion
- `MuPDF <https://mupdf.com/>`_: PDF-to-SVG file conversion

Note: each of these software packages are free except for MSES. However, even MSES
is free by request for academic research.
