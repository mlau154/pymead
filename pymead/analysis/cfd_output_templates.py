"""
Templates to use for aerodynamic forces output of XFOIL and MSES
"""
import typing


class AeroDataOutputTemplates:

    BaseTemplates = {
        "XFOIL": {"Cd": 0.01, "Cl": 0.5, "alf": 0.0, "Cm": 0.01, "Cdf": 0.005, "Cdp": 0.005, "L/D": 50.0, "Cp": {}},
        "MSES": {"Cd": 0.01, "Cl": 0.5, "alf": 0.0, "Cm": 0.01, "Cdf": 0.005, "Cdp": 0.005, "Cdw": 0.005,
                 "Cdv": 0.005, "Cdh": 0.0, "L/D": 50.0, "CPK": 0.1, "BL": []}
    }

    @staticmethod
    def _convert_to_multipoint(template: typing.Dict[str, float], stencil_points: int) -> dict:
        return {k: [v] * stencil_points for k, v in template.items()}

    @staticmethod
    def _convert_to_multigeom(template: typing.Dict[str, float], geoms: int) -> dict:
        return {k: [v] * geoms for k, v in template.items()}

    @staticmethod
    def _convert_to_multigeom_multipoint(template: typing.Dict[str, float], multipoint_active,
                                         stencil_points: typing.List[int]) -> dict:
        return {k: [[v] * n_sp if mp_active and n_sp > 0 else v for n_sp, mp_active in zip(
            stencil_points, multipoint_active)]for k, v in template.items()}

    def get_aero_data_output_template(self, tool: str, geoms: int, multipoint_active: typing.List[bool],
                                      stencil_points: typing.List[int]) -> dict:
        assert len(multipoint_active) == geoms
        base_template = self.BaseTemplates[tool]
        if len(multipoint_active) < 1:
            return base_template
        if len(multipoint_active) == 1:
            if not multipoint_active[0] or stencil_points[0] == 0:
                return base_template
            return self._convert_to_multipoint(base_template, stencil_points=stencil_points[0])
        if all([not mp_active for mp_active in multipoint_active]):
            return self._convert_to_multigeom(base_template, geoms)
        return self._convert_to_multigeom_multipoint(
            base_template, multipoint_active, stencil_points=stencil_points)
