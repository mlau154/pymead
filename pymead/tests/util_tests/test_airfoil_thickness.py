import os

import numpy as np

from pymead import EXAMPLES_DIR
from pymead.core.geometry_collection import GeometryCollection
from pymead.utils.read_write_files import load_data


def test_airfoil_thickness():
    """
    Loads the basic sharp TE airfoil from examples and ensures that its max thickness is equal to the value
    evaluated in the GUI
    """
    geo_col = GeometryCollection.set_from_dict_rep(load_data(os.path.join(EXAMPLES_DIR, "basic_airfoil_sharp.jmea")))
    airfoil = geo_col.container()["airfoils"]["Airfoil-1"]
    data = airfoil.compute_thickness(airfoil_frame_relative=True)
    assert np.isclose(data["t/c_max"], 0.11140262, atol=1e-7)
