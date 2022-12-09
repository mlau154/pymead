=========================
Installation instructions
=========================
Dependencies
============

Required
--------

Each of the following dependencies are required to use **pymead**.

- `scipy <https://scipy.org/>`_: Used for airfoil matching
- `numpy <https://numpy.org/>`_: Used for linear algebra and random numbers
- `shapely <https://shapely.readthedocs.io/en/stable/>`_: Computational geometry
- `matplotlib`: Static plotting
- requests: Downloading airfoil coordinate sets from `Airfoil Tools <http://airfoiltools.com/>`_
- PyQt5: Graphical User Interface (GUI)
- `pyqtgraph`: Interactive plotting and parameter trees
- python-benedict: Dictionary utilities
- pandas: Data structures
- dill: Serialization
- pymoo: Aerodynamic shape optimization
- `numba <https://numba.pydata.org/>`_: Speed-up of inviscid lift coefficient calculation

Optional
--------

**pymead** relies on several external libraries for low- and medium-fidelity
aerodynamic analysis. All the geometry tools in **pymead** are built-in, apart
from several Python libraries that are installed automatically if **pip** is used
as the install method. However, parts of the **analysis** and **optimization** modules
cannot be used without the separate installation of the following external libraries. To
make the most of **pymead**, download the following software packages:

- `XFOIL <https://web.mit.edu/drela/Public/web/xfoil/>`_: low-fidelity,
  single-airfoil-element aerodynamic analysis (linear-strength vortex
  panel code coupled with a boundary-layer model)
- `MSES <https://tlo.mit.edu/technologies/mses-software-high-lift-multielement-airfoil-configurations>`_:
  medium-fidelity, multi-airfoil-element aerodynamic analysis (Euler-equation
  solver coupled with the same boundary-layer model as XFOIL)
- `Ghostscript <https://www.ghostscript.com/>`_: PS-to-PDF file conversion
- `MuPDF <https://mupdf.com/>`_: PDF-to-SVG file conversion

Note: each

Note: each of these software packages are free except for MSES. However, even MSES
is free by request for academic research.
