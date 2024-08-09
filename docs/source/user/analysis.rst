Aerodynamic Analysis
####################

.. |rarrow|   unicode:: U+02192 .. RIGHT ARROW

There are currently three methods for analyzing airfoils in *pymead*: a built-in inviscid linear-strength vortex
panel method, XFOIL, and MSES. Details about these analysis methods are summarized in the table below.

.. |check|   unicode:: U+02705 .. CHECK MARK
.. |cross|   unicode:: U+0274C .. CROSS MARK


.. list-table::
   :widths: 14 16 16 16 20 18
   :header-rows: 1
   :class: max-width-table

   * - Tool
     - Inviscid
     - Viscous
     - Built-in
     - Multi-element
     - Propulsion
   * - :ref:`Panel<panel>`
     - |check|
     - |cross|
     - |check|
     - |cross|
     - |cross|
   * - :ref:`XFOIL<xfoil>`
     - |check|
     - |check|
     - |cross|
     - |cross|
     - |cross|
   * - :ref:`MSES<mses>`
     - |check|
     - |check|
     - |cross|
     - |check|
     - |check|


.. _panel:

Linear-Strength Vortex Panel Method
===================================

The linear-strength vortex panel method built-in to *pymead* is based on Katz & Plotkin [1]. More details about
the theory can be found in the reference, and the API documentation and source code can be found at
:doc:`../_autosummary/pymead.analysis.single_element_inviscid.single_element_inviscid`. Example usages for both the
GUI and API are shown below.

.. note::

    Because this analysis method uses a just-in-time compiler, it will run slowly the first time (though the execution
    time should still be around a few seconds or faster). Subsequent runs should execute significantly faster.

.. tab-set::

    .. tab-item:: GUI
        :sync: gui

        *Construction Zone*

    .. tab-item:: API
        :sync: api

        To analyze an airfoil using the inviscid panel method, only the airfoil coordinates and the angle of attack
        in degrees are required as inputs. Given an airfoil coordinate file called "airfoil_coords.txt" which
        has the airfoil coordinates in the Selig format (counter-clockwise starting from the upper side of the trailing
        edge) and a single header row with the airfoil name, the airfoil can be analyzed as follows:

        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            from pymead.analysis.single_element_inviscid import single_element_inviscid

            coords = np.loadtxt("airfoil_coords.txt", skiprows=1)
            co, cp, cl = single_element_inviscid(coords, alpha=5.0)
            print(f"Lift coefficient = {cl}")
            x = co[:, 0]
            y = co[:, 1]
            plt.plot(x, cp)
            plt.gca().invert_yaxis()
            plt.xlabel(r"$x/c$")
            plt.ylabel(r"$C_p$")
            plt.show()


.. _xfoil:

XFOIL
=====

XFOIL is a vortex panel method coupled with a boundary-layer solver. More information about XFOIL can
be found at the `XFOIL home page <https://web.mit.edu/drela/Public/web/xfoil/>`_.

.. tab-set::

    .. tab-item:: GUI
        :sync: gui

        An XFOIL analysis can be setup and run by navigating in the toolbar to
        **Analysis** |rarrow| **Single-Airfoil** |rarrow| **Viscous**. Selecting this menu option brings up a dialog
        with a number of options. Once the options are configured as desired, press **OK** to run XFOIL. A detailed
        description of the various configuration options is given below.

        .. list-table::
           :widths: 20 80
           :header-rows: 1
           :class: max-width-table

           * - Option
             - Description
           * - Viscosity On?
             - Whether to include the boundary-layer solver in the XFOIL analysis. A separate command is run in XFOIL
               when this option is selected.
           * - Specify Reynolds Number?
             - Whether to directly specify the Reynolds number directly rather than indirectly through Mach number,
               length scale, etc. Selecting this option will disable the option to modify several of the atmospheric
               variables below.
           * - Mach Number
             - The ratio of velocity to speed of sound for the flow. Used along with the temperature, gas constant,
               and specific heat ratio
               to determine the freestream velocity if "Specify Reynolds Number?" is not checked. Also used
               within XFOIL to calculate a compressibility correction to make the analysis more accurate at higher
               Mach numbers. Note that, as the XFOIL documentation mentions, the use of any freestream Mach number where
               supersonic flow over the airfoil occurs will incur severe accuracy penalties.
           * - Specify Flow Variables
             - A combination of two thermodynamic state variables to use to determine the third out of pressure,
               temperature, and density using the ideal gas law.
           * - Pressure (Pa)
             - The static thermodynamic pressure of the airfoil environment in Pascals. Ignored if "Specify Reynolds
               Number?" is checked.
           * - Temperature (K)
             - The static thermodynamic temperature of the airfoil environment in Kelvin. Ignored if "Specify Reynolds
               Number?" is checked.
           * - Density (kg/m^3)
             - The density of the airfoil environment in kilograms per cubic meter. Ignored if "Specify Reynolds
               Number?" is checked.
           * - Specific Heat Ratio
             - The ratio of specific heat at constant pressure to specific heat at constant volume. Ignored if
               "Specify Reynolds Number?" is checked.
           * - Length Scale (m)
             - The length scale, in meters, used to determine the Reynolds number. Ignored if "Specify Reynolds Number?"
               is checked.
           * - Gas Constant (J/(kg*K))
             - The specific gas constant in Joules per kilogram Kelvin. Ignored if "Specify Reynolds Number?" is
               checked.


    .. tab-item:: API
        :sync: api

        *Construction Zone*

.. _mses:

MSES
====


References
==========

[1] J. Katz and A. Plotkin, Low-Speed Aerodynamics, Second Edition, 2nd ed. New York, NY,
    USA: Cambridge University Press, 2004. Accessed: Mar. 07, 2023. [Online].
    Available: `<https://asmedigitalcollection.asme.org/fluidsengineering/article/126/2/293/458666/LowSpeed-Aerodynamics-Second-Edition>`_
