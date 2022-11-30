=========================
Installation instructions
=========================
Dependencies
------------

**pymead** relies on several external libraries for low- and medium-fidelity
aerodynamic analysis. All the geometry tools in **pymead** are built-in, apart
from several Python libraries that are installed automatically if **pip** is used
as the install method. However, parts of the **analysis** and **optimization** modules
cannot be used without the separate installation of the following external libraries:

- `XFOIL <https://web.mit.edu/drela/Public/web/xfoil/>`_: low-fidelity aerodynamic analysis (linear-strength vortex
  panel code coupled with a boundary-layer model)
- `MSES <https://tlo.mit.edu/technologies/mses-software-high-lift-multielement-airfoil-configurations>`_:
  medium-fidelity aerodynamic analysis (Euler-equation solver coupled with the same boundary-layer model as XFOIL)
- `Ghostscript <https://www.ghostscript.com/>`_: PS-to-PDF file conversion
- `MuPDF <https://mupdf.com/>`_: PDF-to-SVG file conversion
