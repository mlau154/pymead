r"""
Welcome
=======
To the documentation page for pymead, an object-oriented Python 3 package for single- and multi-element
Bézier-parametrized airfoil design. This Bézier parametrization framework is being presented at the 2022 AIAA Aviation
Conference in Chicago, IL under the title "A Parametrization Framework for Multi-Element Airfoil Systems Using Bézier
curves."


Motivation
==========
The creation of this package was motivated by a research aircraft application: the aerodynamic design of a
propulsion-airframe-integrated commercial transport aircraft. The cross-section of a wing or fuselage with integrated
propulsors can be represented, with some sacrifice in fidelity, as a quasi-2D multi-element airfoil system. This
multi-element airfoil system comprises a main airfoil (either the fuselage or main airfoil element), a hub
airfoil (representing the cross-section of an axisymmetric hub), and a nacelle airfoil (representing the cross-section
of an axisymmetric nacelle).

By using a well-defined parametrization framework, this airfoil system can be morphed or deformed in a variety of
ways simply by changing the value of the input parameters. These parameters are represented by
``pymead.core.param.Param`` objects in this framework. Defining the airfoil system in this way provides an
intuitive I/O interface for shape optimization or parametric sweeps.

In `pymead`, airfoils are comprised of a set of connected, arbitrary-order Bézier curves. Because Bézier curves have
the property that they always pass through their starting and ending control points, Bézier curve "joints" can be used
to force the airfoil surface to pass through a particular point in space. `pymead` forces all Bézier curve joints
within an airfoil to be :math:`G^0`, :math:`G^1`, and :math:`G^2` continuous, which is useful in general for surface
smoothness and in particular for computational fluid dynamics (CFD) packages where a discontinuity in the curvature
value at a point can cause undesired flow properties or even unconverged results.


Applications
============
It is the hope of the author that pymead is sufficiently flexible to be used for airfoil applications of
varying complexities, from simple, single-airfoil design to high-fidelity, multi-element airfoil shape optimization.
Other common multi-element airfoil systems, such as the high-lift configuration on an aircraft, are also target
applications for this software package.

One utility provided in this software package which may be useful in the start-up phase of airfoil design is
``pymead.utils.airfoil_matching.match_airfoil()``. This function allows the matching of a particular parametrization
to any public airfoil geometry at [airfoiltools.com](http://airfoiltools.com/) or local set of airfoil coordinates
using the gradient-based "SLSQP" optimizer.


Acknowledgments
===============
This work was supported by NASA under award number 80NSSC19M0125 as part of the Center for High-Efficiency Electrical
Technologies for Aircraft (CHEETA). Logo courtesy of [NASA](https://www.nasa.gov/).


Contact Information
===================
**Author**: Matthew G Lauer

**Email**: mlauer2015@gmail.com


Version Notes
=============

1.1.1
-----
- Corrections to README.md for PyPi long project description (images not showing properly)

1.1.0
-----
- Made corrections on BaseAirfoilParams and AnchorPoint Args domains
- Added support for zero-curvature anchor points
  using 180-degree curvature control arm angles (or 90-degree curvature control arm angles for the leading edge)
- Added support for sharp-juncture anchor points with :math:`R=0` or :math:`R_{LE}=0`. Adding multiple consecutive sharp
  juncture anchor points creates line segments. Adding sharp-juncture anchor points violates the principle of slope and
  curvature continuity, but may be useful in some cases.

"""
import os

from PyQt6.QtCore import QCoreApplication, QSettings
import numpy as np

# Monkeypatch for pymoo==0.5.0 (msort deprecation error)
np.msort = np.sort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESOURCE_DIR = os.path.join(BASE_DIR, "resources")
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")
ICON_DIR = os.path.join(BASE_DIR, "icons")
PLUGINS_DIR = os.path.join(BASE_DIR, "plugins")
INCLUDE_FILES = [os.path.join(BASE_DIR, "core", "symmetry.py")]
GUI_DEFAULTS_DIR = os.path.join(BASE_DIR, "gui", "gui_settings", "defaults")
GUI_SETTINGS_DIR = os.path.join(BASE_DIR, "gui", "gui_settings")
GUI_THEMES_DIR = os.path.join(BASE_DIR, "gui", "gui_settings", "themes")
GUI_DEFAULT_AIRFOIL_DIR = os.path.join(BASE_DIR, "gui", "default_airfoil")
GUI_DIALOG_WIDGETS_DIR = os.path.join(BASE_DIR, "gui", "dialog_widgets")
TEST_DIR = os.path.join(BASE_DIR, "tests")

QCoreApplication.setOrganizationName("mlaero")
QCoreApplication.setApplicationName("pymead")

q_settings = QSettings()


class DependencyNotFoundError(Exception):
    pass


class TargetPathNotFoundError(Exception):
    pass


class InvalidFileFormat(Exception):
    pass


def count_lines_of_code():
    """
    Counts the lines of code in each sub-package and prints the results.
    """
    print("Counting the lines of code in all of pymead's sub-packages...")

    def _count_lines(folder: str):
        n_lines = 0
        for file_or_folder in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, file_or_folder)):
                n_lines += _count_lines(os.path.join(folder, file_or_folder))
            else:
                if os.path.splitext(file_or_folder)[-1] != ".py":
                    continue
                n_lines += sum(1 for _ in open(os.path.join(folder, file_or_folder)))
        return n_lines

    total_lines = 0
    for file_path in os.listdir(BASE_DIR):
        if not os.path.isdir(file_path) or file_path == "__pycache__":
            continue
        subpackage_lines = _count_lines(file_path)
        total_lines += subpackage_lines
        print(f"Subpackage {file_path}: {subpackage_lines}")
    print(f"Total Lines: {total_lines}")
