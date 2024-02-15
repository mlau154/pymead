=======
Install
=======

Installation methods
====================

There are several easy ways to install pymead:

Method 1: Windows Installer/Linux Tarball (GUI Only)
----------------------------------------------------

For users merely wishing to use pymead rather than develop pymead, this is the recommended install method.

**Windows**

Go to the `release page on GitHub <https://github.com/mlau154/pymead/releases>`_ and download the ``.exe``
file under the "Assets" dropdown menu. Click on the ``.exe`` and follow the self-contained instructions
to install pymead. For this install method, neither Python nor any of the "required_" dependencies are
necessary to run the pymead executable. Only the "optional_" dependencies are necessary to run
some commands in pymead.

You will be notified automatically at application startup if there is an update for pymead available.
On Windows, the installation wizard will handle the uninstall/upgrade process for you automatically once
it is downloaded and run.

To run pymead, double-click on the pymead program created in the selected install location. Alternatively,
type *pymead* in the Windows search bar and press enter.

**Linux**

Follow similar steps to the procedure for Windows, except download the ``-linux.tar.gz`` file instead of the
``.exe`` file. Then, move the tarball to the desired location and extract it in that location by double-clicking
the tarball in a file explorer or by navigating to the tarball's location and using

.. code-block::

  tar -xvzf <pymead-tarball-name.tar.gz>

in a terminal. The pymead application can then be run directly from inside that extracted folder. When downloading
updates to pymead (you will be notified of these at application startup when they are available), you can
simply replace the original extracted folder with the newly extracted folder. It is *very important* to not remove
the pymead application from the folder that contains the ``_internals`` folder, since pymead needs these to run.

**macOS**

Follow the same steps as Linux, except make sure to download the ``-macOS.tar.gz`` file instead of the Linux version!


Method 2: ``pip`` (GUI + API)
-----------------------------
Use ``pip`` to install the latest stable version of pymead into the environment from the Python Package Index (PyPi):

.. code-block::

  pip install pymead

.. important:: At the moment, you must have a Python version ``>=3.10`` to install pymead using pip.

This method automatically installs all required dependencies that are not yet installed. It also
allows the user to easily update pymead if desired when a new version is available. To update pymead, use

.. code-block::

  pip install pymead --upgrade

The pymead GUI can be started from any directory where pymead is installed by running the following command in the
terminal (use ``python3`` instead of ``py`` for Linux):

.. code-block::

  py -m pymead.gui.gui

Most of the API elements are stored in ``pymead.core``. For example, to create a geometry collection (the main
container in the pymead API), and add a point at :math:`x=0.5`, :math:`y=0.3`, run the following lines in a ``.py``
script or in a Python console:

.. code-block:: python

  from pymead.core.geometry_collection import GeometryCollection
  geo_col = GeometryCollection()
  geo_col.add_point(0.5, 0.3)


Method 3: IDE (GUI + API)
-------------------------
Some IDEs, like `PyCharm <https://www.jetbrains.com/pycharm/>`_, have a plugin for ``pip``. In PyCharm,
simply search for and install "pymead" in the "Python Packages" tab. Follow similar steps as Method 2 for
accessing the GUI and the API. Alternatively, to start the GUI, it is possible to simply click on the
file called ``gui.py`` inside ``pymead/gui``, right-click, and press run.

Method 4: Local Install (DEV: GUI+API)
-----------------------------------------
This method is recommended for those wishing to contribute to pymead in any capacity.
The pymead package can be installed in a local location using `Git <https://gitforwindows.org/>`_.
To accomplish this, first clone the directory using

.. code-block::

  git clone https://github.com/mlau154/pymead.git

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

  import pymead

If the installation was successful, no errors will be thrown.

Dependencies
============

Required
--------

Each of the following dependencies are required to use pymead. All packages listed in this section are automatically
installed when using Methods 1, 2, or 3 above. If using Method 4, the line ``pip install .`` installs these
dependencies.

- `scipy <https://scipy.org/>`_: Used for airfoil matching
- `numpy <https://numpy.org/>`_: Used for math, vector, and matrix computations
- `shapely <https://shapely.readthedocs.io/en/stable/>`_: Computational geometry
- `matplotlib <https://matplotlib.org/>`_: Static plotting
- `requests <https://requests.readthedocs.io/en/latest/>`_: Downloading airfoil coordinate sets
  from `Airfoil Tools <http://airfoiltools.com/>`_
- `PyQt5 <https://pypi.org/project/PyQt5/>`_: Graphical User Interface (GUI)
- `pyqtgraph <https://www.pyqtgraph.org/>`_: Interactive plots
- `python-benedict <https://pypi.org/project/python-benedict/>`_: Dictionary utilities
- `pandas <https://pandas.pydata.org/>`_: Data structures
- `pymoo <https://pymoo.org/>`_: Genetic algorithms used for aerodynamic shape optimization
- `numba <https://numba.pydata.org/>`_: Speed-up of inviscid lift coefficient calculation
- `PyQtWebEngine <https://pypi.org/project/PyQtWebEngine/>`_: Internal GUI web-based help browser
- `cmcrameri <https://www.fabiocrameri.ch/colourmaps/>`_: Perceptually uniform, color-vision-deficiency friendly color
  maps by Fabio Crameri (used for flow visualization)
- `networkx <https://networkx.org/documentation/stable/>`_: Analysis of the undirected graph describing the geometric
  constraint system
- `psutil <https://pypi.org/project/psutil/>`_: Process management

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

Each of these software packages are free except for MSES. However, even MSES
is free by request for academic research. It is important that for each of these programs installed, the full path
to the folder containing the executable be added to the system path. Please see
this blog post at
`medium.com <https://medium.com/@kevinmarkvi/how-to-add-executables-to-your-path-in-windows-5ffa4ce61a53>`_ for more
details on how to accomplish this if you are unfamiliar. As an example, after XFOIL is downloaded from the linked web
page and extracted to the same folder it was downloaded to, a path that looks like
``C:\Users\<user-name>\Downloads\XFOIL6.99`` on Windows is the folder that should be added to the
system path because it contains ``xfoil.exe``. If the XFOIL folder is moved to a more typical
folder used for storing programs, such as ``C:\Program Files``, ``C:\Program Files (x86)``, or
``C:\Users\<user-name>\AppData\Local\Programs`` in Windows, be sure to change the path
accordingly in the environmental variable or Windows will be unable to find the program when
run through pymead.
