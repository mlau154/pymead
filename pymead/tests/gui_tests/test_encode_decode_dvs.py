import os

import numpy as np

from pymead.core.geometry_collection import GeometryCollection
from pymead import EXAMPLES_DIR, TEST_DIR
from pymead.tests.gui_tests.utils import app
from pymead.utils.read_write_files import load_data


def test_iso_prop_decode_gen_10(app):

    # File setup
    alg10_file = os.path.join(TEST_DIR, "gui_tests", "data", "iso_prop_test_0", "algorithm_gen_10.pkl")
    geo_col_file = os.path.join(EXAMPLES_DIR, "isolated_propulsor.jmea")

    def dialog_test_action(dialog):
        """Test action to apply to the dialog. Simply set the value of the dialog and accept."""
        dialog.setModal(False)
        dialog.show()
        dialog.load_airfoil_alg_file_widget.pkl_line.setText(alg10_file)
        dialog.accept()

    # Load the isolated propulsor example
    app.load_example_isolated_propulsor()

    # Load the serialized 10th generation data from the pre-computed optimization
    app.import_algorithm_pkl_file(dialog_test_action=dialog_test_action)

    # Get the new airfoil coordinates list
    coords_list_gui = app.geo_col.container()["mea"]["MEA-1"].get_coords_list()

    # Now, load in the geometry collection and algorithm file directly into memory
    alg10 = load_data(alg10_file)
    geo_col_dict = load_data(geo_col_file)
    X10_api = alg10.opt.get("X")[0, :]
    geo_col = GeometryCollection.set_from_dict_rep(geo_col_dict)

    # Decode the serialized 10th generation data from the pre-computed optimization
    geo_col.assign_design_variable_values(X10_api, bounds_normalized=True)

    # Get the new airfoil coordinates list
    coords_list_api = geo_col.container()["mea"]["MEA-1"].get_coords_list()

    # Make sure that the bounds-normalized list matches the one that was loaded in
    for x_loaded, x_geo_col in zip(X10_api, geo_col.extract_design_variable_values(bounds_normalized=True)):
        assert np.isclose(x_loaded, x_geo_col, rtol=1e-14)

    # Make sure that the value of one of the design variables is equal in both the GUI and API instances of the
    # geometry collection
    angle_2_api = geo_col.container()["desvar"]["Angle-2"].value()
    angle_2_gui = app.geo_col.container()["desvar"]["Angle-2"].value()
    assert np.isclose(angle_2_api, angle_2_gui)

    # Ensure that all the evaluated airfoil coordinate points are equal in both the GUI and API instances of the
    # geometry collection
    for airfoil_coords_api, airfoil_coords_gui in zip(coords_list_api, coords_list_gui):
        for coord_api, coord_gui in zip(airfoil_coords_api, airfoil_coords_gui):
            for x_or_y_api, x_or_y_gui in zip(coord_api, coord_gui):
                assert np.isclose(x_or_y_api, x_or_y_gui, rtol=1e-14)
