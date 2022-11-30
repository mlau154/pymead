=========================
Installation instructions
=========================

**pymead** relies on several external libraries for low- and medium-fidelity
aerodynamic analysis. All the geometry tools in **pymead** are built-in, apart
from several Python libraries that are installed automatically if **pip** is used
as the install method. However, parts of the **analysis** and **optimization** modules
cannot be used without the separate installation of the following external libraries:

- XFOIL: low-fidelity aerodynamic analysis (linear-strength vortex panel code coupled
  with a boundary-layer model)
- MSES: medium-fidelity aerodynamic analysis (Euler-equation solver coupled with the
  same boundary-layer model as XFOIL)
- Ghostscript: PS-to-PDF file conversion
- MuPDF: PDF-to-SVG file conversion
