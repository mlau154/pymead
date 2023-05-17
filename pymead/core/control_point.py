from copy import deepcopy


class ControlPoint:
    """
    Base class for ``pymead.core.anchor_point.AnchorPoint``\ s and ``pymead.core.free_point.FreePoint``\ s.
    """

    def __init__(self, x, y, tag: str, anchor_point_tag: str, cp_type: str = None):
        r"""
        Parameters
        ==========
        x
          ``pymead.core.param.Param`` indicating the :math:`x`-location of the ControlPoint

        y
          ``pymead.core.param.Param`` indicating the :math:`y`-location of the ControlPoint

        tag: str
          Description of the ControlPoint

        anchor_point_tag: str
          Which ``pymead.core.anchor_point.AnchorPoint`` the ControlPoint belongs to

        cp_type: str
          Description of the ControlPoint (one of ``"anchor_point"``, ``"free_point"``, ``"g1_minus"``, ``"g1_plus"``,
          ``"g2_minus"``, ``"g2_plus"``)
        """
        self.x_val = x
        self.y_val = y
        self.anchor_point_tag = anchor_point_tag
        self.tag = tag
        self.xp = deepcopy(self.x_val)
        self.yp = deepcopy(self.y_val)
        self.cp_type = cp_type

    def __repr__(self):
        return f"control_point_{self.tag}"
