import os
import shutil
import time
import warnings

import pytest

from pymead.tests.gui_tests.utils import app
from pymead.utils.read_write_files import load_data
from pymead import TEST_DIR


@pytest.mark.skip(reason="Not working in GitHub Actions")
def test_xfoil_evaluate(app):
    # Make sure that XFOIL is found on the system path. Print a warning if is not.
    if shutil.which("xfoil") is None:
        warnings.warn("XFOIL executable ('xfoil') not found on system path. Skipping this test.")
        return

    # File/directory setup
    base_dir = os.path.join(TEST_DIR, "gui_tests")
    airfoil_name = "test_sharp_xfoil"
    analysis_path = os.path.join(base_dir, airfoil_name)

    def dialog_test_action(dialog):
        """Test action to apply to the dialog. Simply set the value of the dialog and accept."""
        dialog.setModal(False)
        dialog.show()
        dialog.w.widget_dict["base_dir"]["widget"].setText(base_dir)
        dialog.w.widget_dict["airfoil_name"]["widget"].setText(airfoil_name)
        dialog.accept()

    # Load the basic airfoil with sharp trailing edge example
    print(f"Loading the basic airfoil...")
    app.load_example_basic_airfoil_sharp()

    # Run the dialog non-modal and feed it the values loaded into the "mses_settings" variable
    print(f"Running the viscous analysis...")
    aero_data = app.single_airfoil_viscous_analysis(dialog_test_action=dialog_test_action)

    # Ensure that the drag value computed is correct
    assert aero_data["Cd"] == 0.00586

    # If the analysis directory was successfully created, remove it so that the test can work properly when repeated.
    if os.path.exists(analysis_path):
        shutil.rmtree(analysis_path)


@pytest.mark.skip(reason="Not working in GitHub Actions")
def test_mses_evaluate(app):
    """
    Ensures that direct MSES analysis from the GUI is working properly.
    """
    # Make sure that MSET, MSES, and MPLOT are found on the system path. Print a warning if they are not.
    if shutil.which("mset") is None:
        warnings.warn("MSES suite executable 'mset' not found on system path. Skipping this test.")
        return
    if shutil.which("mses") is None:
        warnings.warn("MSES suite executable 'mses' not found on system path. Skipping this test.")
        return
    if shutil.which("mplot") is None:
        warnings.warn("MPLOT suite executable 'mplot' not found on system path. Skipping this test.")
        return

    # File/directory setup
    mses_settings = load_data(os.path.join(TEST_DIR, "gui_tests", "iso_prop_mses_settings.json"))
    mses_settings["MSET"]["airfoil_analysis_dir"] = os.path.join(TEST_DIR, "gui_tests")
    analysis_path = os.path.join(TEST_DIR, "gui_tests", "iso_prop")
    aero_data_path = os.path.join(analysis_path, "aero_data.json")

    def dialog_test_action(dialog):
        """Test action to apply to the dialog. Simply set the value of the dialog and accept."""
        dialog.setModal(False)
        dialog.show()
        dialog.setValue(mses_settings)
        dialog.accept()

    # Load the isolated propulsor example
    app.load_example_isolated_propulsor()

    # Run the dialog non-modal and feed it the values loaded into the "mses_settings" variable
    app.multi_airfoil_analysis_setup(dialog_test_action=dialog_test_action)

    # Because the airfoil system analysis takes a variable amount of time, we need to allow some time for the
    # analysis to complete. Therefore, we check every "t_increment" seconds for the existence of the aero data
    # file and only fail the test if the time given by "t_max" is reached before the file is found.
    t_increment, t_max = 0.1, 15.0
    t = 0.0
    while True:
        try:
            aero_data = load_data(aero_data_path)
            break
        except FileNotFoundError:
            t += t_increment
            if t > t_max:
                raise ValueError("Could not load the aero data in the time allotted")
            time.sleep(t_increment)

    # Ensure that the drag value computed is correct
    assert aero_data["Cd"] == -0.169863740764

    # If the analysis directory was successfully created, remove it so that the test can work properly when repeated.
    if os.path.exists(analysis_path):
        shutil.rmtree(analysis_path)
