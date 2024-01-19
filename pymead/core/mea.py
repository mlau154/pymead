import itertools
import typing
import os

import numpy as np
from pymead.plugins.IGES.iges_generator import IGESGenerator

from pymead.plugins.IGES.curves import BezierIGES

from pymead.core.airfoil import Airfoil
from pymead.core.pymead_obj import PymeadObj


class MEA(PymeadObj):

    def __init__(self, airfoils: typing.List[Airfoil], name: str or None = None):
        self.airfoils = airfoils
        print(f"{self.airfoils = }")
        super().__init__(sub_container="mea")

        # Name the MEA
        name = "MEA-1" if name is None else name
        self._name = None
        self.set_name(name)
        print(f"Now, {self.airfoils = }")

    def add_airfoil(self, airfoil: Airfoil):
        self.airfoils.append(airfoil)

    def remove_airfoil(self, airfoil: Airfoil):
        self.airfoils.remove(airfoil)

    def get_coords_list(self):
        return [airfoil.get_coords_selig_format() for airfoil in self.airfoils]

    def write_mses_blade_file(self,
                              airfoil_sys_name: str,
                              blade_file_dir: str,
                              mea_coords_list: typing.List[np.ndarray] or None = None,
                              grid_bounds: typing.List[float] or None = None
                              ):

        # Get the MEA coordinates list if not provided
        if mea_coords_list is None:
            mea_coords_list = self.get_coords_list()

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
                mea_coords = np.row_stack((mea_coords, np.array([999.0, 999.0])))  # MSES-specific airfoil delimiter
                mea_coords = np.row_stack((mea_coords, airfoil_coords))  # Append this airfoil's coordinates to the mat.

        # Generate the full file path
        blade_file_path = os.path.join(blade_file_dir, f"blade.{airfoil_sys_name}")

        # Save the coordinates to file
        np.savetxt(blade_file_path, mea_coords, header=header, comments="")

        return blade_file_path

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
            for a in self.airfoils.values()]
        entities_flattened = list(itertools.chain.from_iterable(bez_IGES_entities))
        iges_generator = IGESGenerator(entities_flattened)
        iges_generator.generate(file_name)

    def get_dict_rep(self):
        return {"airfoils": [a.name() for a in self.airfoils]}
