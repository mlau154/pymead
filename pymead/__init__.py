"""

## Welcome
To the documentation page for `pymead`, an object-oriented Python 3 package for single- and multi-element
Bézier-parametrized airfoil design. This Bézier parametrization framework is being presented at the 2022 AIAA Aviation
Conference in Chicago, IL under the title "A Parametrization Framework for Multi-Element Airfoil Systems Using Bézier
curves."



## Motivation

The creation of this package was motivated by a research aircraft application: the aerodynamic design of a
propulsion-airframe-integrated commercial transport aircraft. The cross-section of a wing or fuselage with integrated
propulsors can be represented, with some sacrifice in fidelity, as a quasi-2D multi-element airfoil system. This
multi-element airfoil system is comprised of a main airfoil (either the fuselage or main airfoil element), a hub
airfoil (representing the cross-section of an axisymmetric hub), and a nacelle airfoil (representing the cross-section
of an axisymmetric nacelle).



By using a well-defined parametrization framework, this airfoil system can be morphed or deformed in a variety of
ways simply by changing the value of any of the input parameters. These parameters are represented by
`pymead.core.param.Param` objects in this framework. Defining the airfoil system in this way provides an
intuitive I/O interface for shape optimization or parametric sweeps.

In `pymead`, airfoils are comprised of a set of connected, arbitrary-order Bézier curves. Because Bézier curves have
the property that they always pass through their starting and ending control points, Bézier curve "joints" can be used
to force the airfoil surface to pass through a particular point in space. `pymead` forces all Bézier curve joints
within an airfoil to be \\(G^0\\), \\(G^1\\), and \\(G^2\\) continuous, which is useful in general for surface
smoothness and in particular for computational fluid dynamics (CFD) packages where a discontinuity in the curvature
value at a point can cause undesired flow properties or even unconverged results.

## Applications

It is the hope of the author that `pymead` is sufficiently flexible to be used for airfoil applications of
varying complexities, from simple, single-airfoil design to high-fidelity, multi-element airfoil shape optimization.
Other common multi-element airfoil systems, such as the high-lift configuration on an aircraft, are also target
applications for this software package.



One utility provided in this software package which may be useful in the start-up phase of airfoil design is
`pymead.utils.airfoil_matching.match_airfoil()`. This modules allows the matching of a particular parametrization
to any public airfoil geometry at [airfoiltools.com](http://airfoiltools.com/) using the gradient-based "SLSQP"
optimizer.



## Acknowledgments

This work was supported by NASA under award number 80NSSC19M0125 as part of the Center for High-Efficiency Electrical
Technologies for Aircraft (CHEETA). Logo courtesy of [NASA](https://www.nasa.gov/).



## Contact Information

**Author**: Matthew G Lauer

**Email**: mlauer2015@gmail.com

## Version Notes

### 1.1.1

- Corrections to README.md for PyPi long project description (images not showing properly)

### 1.1.0

- Made corrections on BaseAirfoilParams and AnchorPoint Args domains
- Added support for zero-curvature anchor points
using 180-degree curvature control arm angles (or 90-degree curvature control arm angles for the leading edge)
- Added support for sharp-juncture anchor points with \(R=0\) or \\(R_{LE}=0\\). Adding multiple consecutive sharp
juncture anchor points creates line segments. Adding sharp-juncture anchor points violates the principle of slope and
curvature continuity, but may be useful in some cases.

"""
import os
from matplotlib import colormaps
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
PLUGINS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")
INCLUDE_FILES = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "symmetry.py")]
GUI_DEFAULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui", "gui_settings", "defaults")
