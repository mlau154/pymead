import typing

from pymead.core.param import Param
from pymead.core.pos_param import PosParam
from pymead.core.anchor_point import AnchorPoint
from pymead.core.trailing_edge_point import TrailingEdgePoint


class BaseAirfoilParams:

    def __init__(self,
                 airfoil_tag: str = None,
                 c: Param = None,
                 alf: Param = None,
                 R_le: Param = None,
                 L_le: Param = None,
                 r_le: Param = None,
                 phi_le: Param = None,
                 psi1_le: Param = None,
                 psi2_le: Param = None,
                 L1_te: Param = None,
                 L2_te: Param = None,
                 theta1_te: Param = None,
                 theta2_te: Param = None,
                 t_te: Param = None,
                 r_te: Param = None,
                 phi_te: Param = None,
                 dx: Param = None,
                 dy: Param = None,
                 ):
        r"""
        The most fundamental parameters required for the generation of any ``pymead.core.airfoil.Airfoil``.
        A geometric description of an example airfoil generated (from ``pymead.examples.simple_airfoil.run()``) is
        shown below (it may be helpful to open the image in a new tab to adequately view the details):

        Parameters
        ==========
        airfoil_tag: str
          The Airfoil to which this set of base parameters belongs.

        c: Param or None
          :math:`c`: Chord length. Default value if ``None``: ``Param(1.0)``.

        alf: Param or None
          :math:`\alpha`: Angle of attack [rad]. Default value if ``None``: ``Param(0.0)``.

        R_le: Param or None
          :math:`R_{LE}`: Leading-edge radius. Default value if ``None``: ``Param(0.1)``.

        L_le: Param or None
          :math:`L_{LE}`: Distance between the control points immediately before and after the leading-edge
          anchor point. Default value if ``None``: ``Param(0.1)``.

        r_le: Param or None
          :math:`r_{LE}`: Ratio of the distance from the leading-edge anchor point to the control point before to
          the distance between the control points immediately before and after the leading-edge
          anchor point (:math:`r_{LE} = L_{LE,\text{upper}} / L_{LE}`). Default value if ``None``: ``Param(0.5)``.

        phi_le: Param or None
          :math:`\phi_{LE}`: Leading-edge tilt (rad), referenced counter-clockwise from the perpendicular to
          the chordline. Default value if ``None``: ``Param(0.0)``.

        psi1_le: Param or None
          :math:`\psi_{LE,1}`: Leading-edge upper curvature control angle (rad), referenced counter-clockwise
          from the chordline: Default value if ``None``: ``Param(0.0)``.

        psi2_le: Param or None
          :math:`\psi_{LE,2}`: Leading-edge lower curvature control angle (rad), referenced clockwise
          from the chordline. Default value if ``None``: ``Param(0.0)``.

        L1_te: Param or None
          :math:`L_{TE,1}`: Trailing edge upper length. Default value if ``None``: ``Param(0.1)``.

        L2_te: Param or None
          :math:`L_{TE,2}`: Trailing edge lower length. Default value if ``None``: ``Param(0.1)``.

        theta1_te: Param or None
          :math:`\theta_{TE,1}`: Trailing edge upper angle (rad), referenced clockwise from the chordline.
          Default value if ``None``: ``Param(np.deg2rad(10.0))``.

        theta2_te: Param or None
          :math:`\theta_{TE,2}`: Trailing edge lower angle (rad), referenced counter-clockwise from the
          chordline. Default value if ``None``: ``Param(np.deg2rad(10.0))``.

        t_te: Param or None
          :math:`t_{TE}`: Blunt trailing edge thickness.
          Default value if ``None``: ``Param(0.0)`` (sharp trailing edge).

        r_te: Param or None
          :math:`r_{TE}`: Ratio of the distance from the chordline's endpoint at the trailing edge to the
          first control point of the airfoil to the distance between the first and last control points of the airfoil
          (:math:`r_{TE} = L_{TE,upper} / L_{TE}`). This parameter has no effect
          on the airfoil geometry unless ``t_te != Param(0.0)``. Default value if ``None``: ``Param(0.5)``.

        phi_te: Param or None
          :math:`\phi_{TE}`: Blunt trailing-edge tilt (rad), referenced counter-clockwise from the
          perpendicular to the chordline (same as ``phi_le``). This parameter has no effect
          on the airfoil geometry unless ``t_te != Param(0.0)``. Default value if ``None``: ``Param(0.0)``.

        dx: Param or None
          :math:`\Delta x`: Distance to translate the airfoil in the :math:`x`-direction. The translation operation
          follows the rotation operation such that the rotation operation can be performed about the origin.
          Default value if ``None``: ``Param(0.0)``.

        dy: Param or None
          :math:`\Delta y`: Distance to translate the airfoil in the :math:`y`-direction. The translation operation
          follows the rotation operation such that the rotation operation can be performed about the origin.
          Default value if ``None``: ``Param(0.0)``.

        Returns
        =======
        BaseAirfoilParams
          An instance of the ``BaseAirfoilParams`` class.
        """
        self.airfoil_tag = airfoil_tag
        self.c = c
        self.alf = alf
        self.R_le = R_le
        self.L_le = L_le
        self.r_le = r_le
        self.phi_le = phi_le
        self.psi1_le = psi1_le
        self.psi2_le = psi2_le
        self.L1_te = L1_te
        self.L2_te = L2_te
        self.theta1_te = theta1_te
        self.theta2_te = theta2_te
        self.t_te = t_te
        self.r_te = r_te
        self.phi_te = phi_te
        self.dx = dx
        self.dy = dy

        if not self.c:
            self.c = Param(1.0)
        if not self.alf:
            self.alf = Param(0.0, periodic=True)
        if not self.R_le:
            self.R_le = Param(0.04)
        if not self.L_le:
            self.L_le = Param(0.1)
        if not self.r_le:
            self.r_le = Param(0.5)
        if not self.phi_le:
            self.phi_le = Param(0.0, periodic=True)
        if not self.psi1_le:
            self.psi1_le = Param(0.3, periodic=True)
        if not self.psi2_le:
            self.psi2_le = Param(0.1, periodic=True)
        if not self.L1_te:
            self.L1_te = Param(0.4)
        if not self.L2_te:
            self.L2_te = Param(0.3)
        if not self.theta1_te:
            self.theta1_te = Param(0.1, periodic=True)
        if not self.theta2_te:
            self.theta2_te = Param(0.1, periodic=True)
        if not self.t_te:
            self.t_te = Param(0.0)
        if not self.r_te:
            self.r_te = Param(0.5)
        if not self.phi_te:
            self.phi_te = Param(0.0, periodic=True)
        if not self.dx:
            self.dx = Param(0.0)
        if not self.dy:
            self.dy = Param(0.0)

    def generate_main_anchor_points(self) -> typing.List[AnchorPoint]:
        """
        Generates the minimal set of ``pymead.core.anchor_point.AnchorPoint``\ s required for an Airfoil in pymead.

        Returns
        =======
        typing.List[AnchorPoint]
        """
        le_anchor_point = AnchorPoint(PosParam((0.0, 0.0)), 'le', 'te_1', self.airfoil_tag, self.L_le, self.R_le,
                                      self.r_le, self.phi_le, self.psi1_le, self.psi2_le)
        te_1_anchor_point = TrailingEdgePoint(self.c, self.r_te, self.t_te, self.phi_te, self.L1_te, self.theta1_te,
                                              True)
        te_2_anchor_point = TrailingEdgePoint(self.c, self.r_te, self.t_te, self.phi_te, self.L2_te, self.theta2_te,
                                              False)
        return [te_1_anchor_point, le_anchor_point, te_2_anchor_point]
