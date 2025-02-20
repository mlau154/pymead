"""
This tutorial shows how the Pymead API can be used to run an MSES analysis on a NACA 23012 airfoil (automatically
downloaded from the web). See the code below or by
clicking on ``main`` under "Functions" at the bottom and then clicking [source] next to the ``main`` function.

.. highlight:: python
.. code-block:: python

    import os
    import tempfile
    from pathlib import PureWindowsPath

    from pymead.analysis.calc_aero_data import (calculate_aero_data, MSETSettings, MSESSettings, MPLOTSettings,
                                                AirfoilMSETMeshingParameters)
    from pymead.core.geometry_collection import GeometryCollection


    def main():
        # Create a geometry collection
        geo_col = GeometryCollection()

        # Download the airfoil and create an Airfoil object from it
        polyline = geo_col.add_polyline(source="naca23012-il")  # "source" can also be a file path
        airfoil = polyline.add_polyline_airfoil()

        # Create an MEA object from the airfoil (required for MSES analysis)
        mea = geo_col.add_mea([airfoil])

        # Configure the MSET settings (see the API reference for more options)
        mset_settings = MSETSettings(
            multi_airfoil_grid={"Airfoil-1": AirfoilMSETMeshingParameters()},
            airfoil_side_points=180
        )

        # Configure the MSES settings (see the API reference for more options)
        mses_settings = MSESSettings(
            xtrs={"Airfoil-1": [1.0, 1.0]},  # This implies free transition;
            # use values between 0.0 and 1.0 for forced transition
            Ma=0.1,
            Re=3.0e5,
            Cl=0.7,
            alfa_Cl_mode=1,  # Should be set to 0 if "alfa" is specified instead
            timeout=25.0  # The timeout for MSES in seconds
        )

        # Configure the MPLOT settings (see the API reference for more options)
        mplot_settings = MPLOTSettings(
            Tecplot=True  # Export the flow field to the Tecplot ASCII .dat file format
        )

        print(f"Running MSES...")
        aero_data, logs = calculate_aero_data(
            conn=None,  # Required argument
            airfoil_coord_dir=tempfile.gettempdir(),  # This can be changed to any directory
            airfoil_name="naca23012",  # This is the name of the analysis directory that gets created
            mea=mea,
            tool="MSES",  # Must set this value since "XFOIL" is the default
            mset_settings=mset_settings,
            mses_settings=mses_settings,
            mplot_settings=mplot_settings,
            export_Cp=True,  # This can be set to False if the pressure distributions are not needed
            save_aero_data=True,  # Setting this option to true exports a JSON file with the aerodynamic
            # data available in the return statement
        )

        # Print a link to the output directory
        analysis_dir = PureWindowsPath(os.path.normpath(os.path.abspath(os.path.dirname(logs['mset'])))).as_posix()
        print(f"Output directory: file:///{analysis_dir}")

        # Print the data
        if not aero_data["converged"]:
            raise ValueError("MSES did not converge")
        if aero_data["errored_out"]:
            raise ValueError("MSES errored out")
        if aero_data["timed_out"]:
            raise ValueError("MSES timed out")
        print(f"Cl: {aero_data['Cl']:.2f} | Cd: {aero_data['Cd'] * 1e4:.2f} counts | Cm: {aero_data['Cm']:.3f}")


    if __name__ == "__main__":
        main()

"""
import os
import tempfile
from pathlib import PureWindowsPath

from pymead.analysis.calc_aero_data import (calculate_aero_data, MSETSettings, MSESSettings, MPLOTSettings,
                                            AirfoilMSETMeshingParameters)
from pymead.core.geometry_collection import GeometryCollection


def main():
    # Create a geometry collection
    geo_col = GeometryCollection()

    # Download the airfoil and create an Airfoil object from it
    polyline = geo_col.add_polyline(source="naca23012-il")  # "source" can also be a file path
    airfoil = polyline.add_polyline_airfoil()

    # Create an MEA object from the airfoil (required for MSES analysis)
    mea = geo_col.add_mea([airfoil])

    # Configure the MSET settings (see the API reference for more options)
    mset_settings = MSETSettings(
        multi_airfoil_grid={"Airfoil-1": AirfoilMSETMeshingParameters()},
        airfoil_side_points=180
    )

    # Configure the MSES settings (see the API reference for more options)
    mses_settings = MSESSettings(
        xtrs={"Airfoil-1": [1.0, 1.0]},  # This implies free transition;
        # use values between 0.0 and 1.0 for forced transition
        Ma=0.1,
        Re=3.0e5,
        Cl=0.7,
        alfa_Cl_mode=1,  # Should be set to 0 if "alfa" is specified instead
        timeout=25.0  # The timeout for MSES in seconds
    )

    # Configure the MPLOT settings (see the API reference for more options)
    mplot_settings = MPLOTSettings(
        Tecplot=True  # Export the flow field to the Tecplot ASCII .dat file format
    )

    print(f"Running MSES...")
    aero_data, logs = calculate_aero_data(
        conn=None,  # Required argument
        airfoil_coord_dir=tempfile.gettempdir(),  # This can be changed to any directory
        airfoil_name="naca23012",  # This is the name of the analysis directory that gets created
        mea=mea,
        tool="MSES",  # Must set this value since "XFOIL" is the default
        mset_settings=mset_settings,
        mses_settings=mses_settings,
        mplot_settings=mplot_settings,
        export_Cp=True,  # This can be set to False if the pressure distributions are not needed
        save_aero_data=True,  # Setting this option to true exports a JSON file with the aerodynamic
        # data available in the return statement
    )

    # Print a link to the output directory
    analysis_dir = PureWindowsPath(os.path.normpath(os.path.abspath(os.path.dirname(logs['mset'])))).as_posix()
    print(f"Output directory: file:///{analysis_dir}")

    # Print the data
    if not aero_data["converged"]:
        raise ValueError("MSES did not converge")
    if aero_data["errored_out"]:
        raise ValueError("MSES errored out")
    if aero_data["timed_out"]:
        raise ValueError("MSES timed out")
    print(f"Cl: {aero_data['Cl']:.2f} | Cd: {aero_data['Cd'] * 1e4:.2f} counts | Cm: {aero_data['Cm']:.3f}")


if __name__ == "__main__":
    main()
