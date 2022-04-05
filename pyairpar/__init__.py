"""

## Welcome
To the documentation page for `pyairpar`, an object-oriented Python 3 package for single- and multi-element
Bézier-parametrized airfoil design.

.. image:: complex_airfoil-3.png

## Motivation

The creation of this package was motivated by a research aircraft application: the aerodynamic design of a
propulsion-airframe-integrated commercial transport aircraft. The cross-section of a wing or fuselage with integrated
propulsors can be represented, with some sacrifice in fidelity, as a quasi-2D multi-element airfoil system. This
multi-element airfoil system is comprised of a main airfoil (either the fuselage or main airfoil element), a hub
airfoil (representing the cross-section of an axisymmetric hub), and a nacelle airfoil (representing the cross-section
of an axisymmetric nacelle).

By using a well-defined parametrization framework, this airfoil system can be morphed or deformed in a variety of
ways simply by changing the value of any of the input parameters. These parameters are represented by
`pyairpar.core.param.Param` objects in this framework. Defining the airfoil system in this way provides an
intuitive I/O interface for shape optimization or parametric sweeps.

In `pyairpar`, airfoils are comprised of a set of connected, arbitrary-order Bézier curves. Because Bézier curves have
the property that they always pass through their starting and ending control points, Bézier curve "joints" can be used
to force the airfoil surface to pass through a particular point in space. `pyairpar` forces all Bézier curve joints
within an airfoil to be \\(G^0\\), \\(G^1\\), and \\(G^2\\) continuous, which is useful in general for surface
smoothness and in particular for computational fluid dynamics (CFD) packages where a discontinuity in the curvature
value at a point can cause undesired flow properties or even unconverged results.

## Applications

It is the hope of the author that `pyairpar` is sufficiently flexible to be used for airfoil applications of
varying complexities, from simple, single-airfoil design to high-fidelity, multi-element airfoil shape optimization.
Other common multi-element airfoil systems, such as the high-lift configuration on an aircraft, are also target
applications for this software package.

One utility provided in this software package which may be useful in the start-up phase of airfoil design is
`pyairpar.utils.airfoil_matching.match_airfoil()`. This modules allows the matching of a particular parametrization
to any public airfoil geometry on [airfoiltools.com](http://airfoiltools.com/) using the gradient-based "SLSQP"
optimizer.

## Contact Information

**Author**: Matthew G Lauer

**Email**: mlauer2015@gmail.com

"""