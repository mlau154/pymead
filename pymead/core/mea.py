import itertools
import os
import typing

import numpy as np
from pymead.utils.read_write_files import save_data

from pymead.core.transformation import Transformation2D
from pymead.core.airfoil import Airfoil
from pymead.core.pymead_obj import PymeadObj
from pymead.plugins.IGES.curves import BezierIGES
from pymead.plugins.IGES.iges_generator import IGESGenerator


class MEA(PymeadObj):

    def __init__(self, airfoils: typing.List[Airfoil], name: str or None = None, geo_col=None):
        self.airfoils = airfoils
        self.reference_airfoil = self.airfoils[0]
        super().__init__(sub_container="mea", geo_col=geo_col)

        # Name the MEA
        name = "MEA-1" if name is None else name
        self._name = None
        self.set_name(name)

    def add_airfoil(self, airfoil: Airfoil):
        self.airfoils.append(airfoil)

    def remove_airfoil(self, airfoil: Airfoil):
        self.airfoils.remove(airfoil)

    def get_coords_list(self, max_airfoil_points: int = None, curvature_exp: float = 2.0):
        mea_coords_list = [
            airfoil.get_coords_selig_format(max_airfoil_points, curvature_exp) for airfoil in self.airfoils]
        max_y = [np.max(coords[:, 1]) for coords in mea_coords_list]
        airfoil_order = np.argsort(max_y)[::-1]
        return [mea_coords_list[a_idx] for a_idx in airfoil_order]

    def get_coords_list_chord_relative(self, max_airfoil_points: int = None,
                                       curvature_exp: float = 2.0):
        coords_list = self.get_coords_list(max_airfoil_points=max_airfoil_points, curvature_exp=curvature_exp)

        # Get the transformation object
        chord_length = self.reference_airfoil.measure_chord()
        transformation_kwargs = dict(
            tx=[0.0], ty=[0.0], r=[0.0], sx=[1 / chord_length], sy=[1 / chord_length],
            rotation_units="rad", order="t,s,r"
        )
        transformation = Transformation2D(**transformation_kwargs)
        transformation_kwargs["length_unit"] = self.geo_col.units.current_length_unit()

        return [transformation.transform(coords) for coords in coords_list], transformation_kwargs

    def write_mses_blade_file(self,
                              airfoil_sys_name: str,
                              blade_file_dir: str,
                              mea_coords_list: typing.List[np.ndarray] or None = None,
                              grid_bounds: typing.List[float] or None = None,
                              max_airfoil_points: int = None, curvature_exp: float = 2.0) -> (str, typing.List[str]):

        # Get the MEA coordinates list if not provided
        if mea_coords_list is None:
            mea_coords_list, transformation_kwargs = self.get_coords_list_chord_relative(
                max_airfoil_points=max_airfoil_points, curvature_exp=curvature_exp)
            save_data(transformation_kwargs, os.path.join(blade_file_dir, "transformation.json"))

        # Set the default grid bounds value
        if grid_bounds is None:
            grid_bounds = [-5.0, 5.0, -5.0, 5.0]

        # Write the header (line 1: airfoil name, line 2: grid bounds values separated by spaces)
        header = airfoil_sys_name + "\n" + " ".join([str(gb) for gb in grid_bounds])

        # Determine the correct ordering for the airfoils. MSES expects airfoils to be ordered from top to bottom
        max_y = [np.max(coords[:, 1]) for coords in mea_coords_list]
        airfoil_order = np.argsort(max_y)[::-1]

        # Loop through the airfoils in the correct order
        mea_coords = None
        for airfoil_idx in airfoil_order:
            airfoil_coords = mea_coords_list[airfoil_idx]  # Extract the airfoil coordinates for this airfoil
            if mea_coords is None:
                mea_coords = airfoil_coords
            else:
                mea_coords = np.vstack((mea_coords, np.array([999.0, 999.0])))  # MSES-specific airfoil delimiter
                mea_coords = np.vstack((mea_coords, airfoil_coords))  # Append this airfoil's coordinates to the mat.

        # Generate the full file path
        blade_file_path = os.path.join(blade_file_dir, f"blade.{airfoil_sys_name}")

        # Save the coordinates to file
        np.savetxt(blade_file_path, mea_coords, header=header, comments="")

        # Get the airfoil name order
        airfoil_name_order = [airfoil.name() for airfoil in [self.airfoils[idx] for idx in airfoil_order]]

        return blade_file_path, airfoil_name_order

    def write_to_IGES(self, file_name: str):
        """
        Writes the airfoil system to file using the IGES file format.

        Parameters
        ==========
        file_name: str
            Path to IGES file
        """
        bez_IGES_entities = [
            [BezierIGES(np.column_stack((c.P[:, 0], np.zeros(len(c.P)), c.P[:, 1]))) for c in a.curve_list]
            for a in self.airfoils]
        entities_flattened = list(itertools.chain.from_iterable(bez_IGES_entities))
        iges_generator = IGESGenerator(entities_flattened)
        iges_generator.generate(file_name)

    def get_max_x_extent(self):
        coords_list, _ = self.get_coords_list_chord_relative()
        max_x_list = [max(coords[:, 0]) for coords in coords_list]
        return max(max_x_list)

    def get_dict_rep(self):
        return {"airfoils": [a.name() for a in self.airfoils]}
