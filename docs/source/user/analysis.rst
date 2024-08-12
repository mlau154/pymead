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
   * - :ref:`MSES<mses-suite>`
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
           * - Prescribe α/Cl/CLI
             - Whether to prescribe angle of attack, viscous lift coefficient, or inviscid lift coefficient.
               If the lift coefficient is prescribed, XFOIL uses the linear lift-curve slope to compute the angle of
               attack required to achieve the prescribed lift coefficient.
           * - Angle of Attack (deg)
             - The angle of attack of the airfoil in degrees. The angle of attack is relative to the angle of the
               input geometry as shown in the geometry canvas, so the total angle of attack analyzed is the sum of
               the two angles of attack.
           * - Viscous Cl
             - Viscous lift coefficient.
           * - Inviscid Cl
             - Inviscid lift coefficient.
           * - Transition x/c (upper)
             - Chord-normalized x-location along the upper surface where transition is forced. Transition can naturally
               occur upstream of the specified location, but it will never occur downstream. If a value of 1.0
               is specified, free transition is allowed.
           * - Transition x/c (lower)
             - Chord-normalized x-location along the lower surface where transition is forced.
           * - Turbulence (NCrit)
             - Transition amplification factor. A value of 9.0 is used for an average wind tunnel. See the
               `XFOIL documentation page <https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt>`_ for details
               about this variable and typical values for other scenarios.
           * - Maximum Iterations
             - The number of iterations allowed during viscous analysis. Ignored if "Viscosity On?" is not checked.
           * - Timeout (sec)
             - The amount of time allotted to an XFOIL analysis. The XFOIL process will be automatically terminated
               after this amount of time regardless of whether the analysis has completed.
           * - Airfoil to Analyze
             - This is the name of an airfoil found in the "Airfoils" container of the parameter tree; default names
               are "Airfoil-1", "Airfoil-2", etc. See the :ref:`airfoils` section to learn about airfoil creation.
           * - Analysis Base Directory
             - This is the directory where a new sub-directory named using the next field ("Airfoil Name") will
               be created to store the analysis files.
           * - Airfoil Name
             - Separate from the "Airfoil to Analyze" option, this is the name given to the analysis sub-directory
               and to several of the files used for analysis.


    .. tab-item:: API
        :sync: api

        *Construction Zone*

.. _mses-suite:

MSES Suite
==========

Because
`MSES <https://tlo.mit.edu/industry-entrepreneurs/available-technologies/mses-software-high-lift-multielement-airfoil>`_
is a full suite of tools/executables rather than a single primary executable like XFOIL, the
field entry descriptions will be split up into several categories that correspond with the vertical tabs in the
MSES analysis dialog. These are MSET (grid generation), MSES (flow analysis), MPLOT (post-processing), and
MPOLAR (polar analysis). Additional information about each of these programs can be found in the
`MSES user guide <https://web.mit.edu/drela/Public/web/mses/mses.pdf>`_.

MSET
----

MSET is built-in grid generation tool within the MSES suite. Note that the MSET tool automatically re-meshes
the airfoil surfaces according to the set of input parameters.

.. tab-set::

    .. tab-item:: GUI
        :sync: gui

        The MSET settings can be accessed from the GUI by navigating in the toolbar to
        **Analysis** |rarrow| **Multi-Element Airfoil**. Descriptions of the various parameters are given below.
        More details on these parameters can be found in the
        `MSES user guide <https://web.mit.edu/drela/Public/web/mses/mses.pdf>`_ in the "MSET" section.

        .. list-table::
           :widths: 20 80
           :header-rows: 1
           :class: max-width-table

           * - Option
             - Description
           * - MEA
             - The name of the multi-element airfoil to be analyzed that matches a name under the
               "Multi-Element Airfoils" sub-container of the parameter tree. Note that even for single-airfoil
               analysis in MSES, a multi-element airfoil must be created. See :ref:`multi-element-airfoils` for
               more information about how to create these objects.
           * - Grid Bounds
             - These four values represent the chord-normalized locations of the four sides of the pseudo-rectangular
               grid boundary. The "Left" and "Right" fields represent the :math:`x/c`-locations of the vertical
               inlet and outlet lines, respectively. The "Bottom" and "Top" fields represent the
               :math:`y/c`-locations of the "floor" and "ceiling" of the flow volume, respectively. Note that all
               MSES analyses automatically normalize the airfoil coordinates by the chord length of the first airfoil
               in the multi-element airfoil system.
           * - Airfoil Side Points
             - The number of grid points allocated to each airfoil side when re-meshing the airfoil surfaces.
           * - Side Points Exponent
             - If this value is 1.0, each airfoil element is allocated a number of grid points proportional to its
               chord. If this value is 0.0, each airfoil receives the same number of grid points.
           * - Inlet Points Left
             - Number of streamwise cells upstream of the leftmost airfoil stagnation point.
           * - Outlet Points Right
             - Number of streamwise cells downstream of the rightmost airfoil stagnation point.
           * - Number Top Streamlines
             - Number of stream-normal cells above the uppermost airfoil surface.
           * - Number Bottom Streamlines
             - Number of stream-normal cells below the lowermost airfoil surface.
           * - Max Streamlines Between
             - Maximum number of stream-normal cells between any two airfoil elements.
           * - Elliptic Parameter
             - Used in the MSET ``gridpar`` file, must does not seem to have an impact on the grid.
           * - Stag. Pt. Aspect Ratio
             - Aspect ratio of the cells at the stagnation points.
           * - X-Spacing Parameter
             - From the `MSES user guide <https://web.mit.edu/drela/Public/web/mses/mses.pdf>`_:
               "...how much the quasi-normal grid lines spread out away from the airfoil.
               A larger value (0.8 ... 1.0) tends to make the grid more orthogonal (which is good), but may cause
               excessive bunching of the quasi-normal grid lines in high-lift cases (which is bad). Transonic flows
               are best run with a nearly-orthogonal grid, since grid shearing increases the amount of dissipation
               needed for stability."
           * - Streamline Gen. Alpha
             - Angle of attack in degrees used to generate the initial set of streamlines.
           * - MSET Timeout
             - This is the maximum amount of time for allotted grid generation, used to prevent hanging grid-generation
               processes from permanently freezing *pymead*.
           * - dsLE/dsAvg
             - Leading edge spacing ratio. Smaller values (closer to 0) refine the surface grid near the leading edge.
           * - dsTE/dsAvg
             - Trailing edge spacing ratio. Smaller values (closer to 0) refine the surface grid near the trailing edge.
           * - Curvature Exponent
             - Large values of this number (> 1) correspond to very fine meshing in areas of high curvature,
               while small values of this number (close to 0) correspond to nearly uniform arc length between
               airfoil surface points.
           * - U_s_smax_min
             - Arc length fraction of the starting position of the upper surface refinement.
           * - U_s_smax_max
             - Arc length fraction of the ending position of the upper surface refinement.
           * - L_s_smax_min
             - Arc length fraction of the starting position of the lower surface refinement.
           * - L_s_smax_max
             - Arc length fraction of the ending position of the lower surface refinement.
           * - U Local/Max. Density Ratio
             - Upper side spacing refinement. Higher values increase the refinement in the region specified
               by "U_s_smax_min" and "U_s_smax_max." A value of 0.0 indicates no refinement.
           * - L Local/Max. Density Ratio
             - Lower side spacing refinement. Higher values increase the refinement in the region specified
               by "L_s_smax_min" and "L_s_smax_max." A value of 0.0 indicates no refinement.
           * - Analysis Directory
             - This is the directory where a new sub-directory named using the next field
               ("Airfoil Coord. Filename") will be created to store the analysis files.
           * - Airfoil Coord. Filename
             - This is the name given to the analysis sub-directory and to several of the files used for analysis.
           * - Save As
             - Save all the information in every tab of this dialog to a ``.json`` settings file.
           * - Load
             - Loading MSES settings from a ``.json`` settings file (saved using the above button).
           * - Use downsampling?
             - Whether to downsample the airfoil system prior to sending to MSET. It is sometimes necessary to check
               this box when a large number of curves is used for an airfoil; this can cause the internal grid
               size limits within MSES to be reached. Downsampling can prevent this error from occurring.
               More details about the downsampling method can be found in the ``downsample`` method of
               :doc:`../_autosummary/pymead.core.airfoil.Airfoil`.
           * - Max downsampling points
             - The maximum number of points to allow for each airfoil.
           * - Downsampling curvature exponent
             - Values close to 0 place high emphasis on curvature, while values close to ∞ place low emphasis on
               curvature (creating nearly uniform spacing).

    .. tab-item:: API
        :sync: api

        *Construction Zone*


MSES
----

MSES is the flow analysis tool in the MSES suite.


.. tab-set::

    .. tab-item:: GUI
        :sync: gui

        The MSES settings can be accessed from the GUI by navigating in the toolbar to
        **Analysis** |rarrow| **Multi-Element Airfoil** and selecting the "MSES" tab.
        Descriptions of the various parameters are given below.
        More details on these parameters can be found in the
        `MSES user guide <https://web.mit.edu/drela/Public/web/mses/mses.pdf>`_.

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
           * - Specify Alpha/Cl
             - Whether to prescribe angle of attack or lift coefficient.
               If the lift coefficient is prescribed, MSES uses the linear lift-curve slope to compute the angle of
               attack required to achieve the prescribed lift coefficient.
           * - Angle of Attack (deg)
             - The angle of attack of the airfoil in degrees. The angle of attack is relative to the angle of the
               input geometry as shown in the geometry canvas, so the total angle of attack analyzed is the sum of
               the two angles of attack.
           * - Lift Coefficient
             - This value will be used to indirectly set the angle of attack.
           * - Isentropic/Momentum
             - Where to use the isentropic condition or the momentum equation. More details about this parameter
               can be found in the `MSES user guide <https://web.mit.edu/drela/Public/web/mses/mses.pdf>`_.
           * - Far-Field Boundary
             - Which boundary condition to use in the far field. More details about the boundary conditions can be
               be found in the `MSES user guide <https://web.mit.edu/drela/Public/web/mses/mses.pdf>`_.
           * - Crit. Amp. Factor
             - Critical amplification factor. A value of 9.0 is used for an average wind tunnel. Same as the
               "Turbulence (NCrit)" option in the XFOIL dialog. See the
               `XFOIL documentation page <https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt>`_ for details
               about this variable and typical values for other scenarios.
           * - Critical Mach Number
             - Mach number where shock waves are triggered. For incompressible cases, 1.0 is a fine value for this
               parameter. For higher Mach number transonic cases, this value may need to be reduced to achieve
               convergence.
           * - Artificial Dissipation
             - The "MUCON" parameter in MSES. For incompressible cases, 1.0 is a fine value for this parameter.
               For higher Mach number transonic cases where significant supersonic regions are present, this
               value may need to be increased to achieve convergence.
           * - Timeout (sec)
             - The amount of time allotted to an MSES analysis. The MSES process will be automatically terminated
               after this amount of time regardless of whether the analysis has completed.
           * - Maximum Iterations
             - The number of iterations allowed during viscous analysis.
           * - XTRSupper
             - Chord-normalized x-location along the upper surface where transition is forced. Transition can naturally
               occur upstream of the specified location, but it will never occur downstream. If a value of 1.0
               is specified, free transition is allowed.
           * - XTRSlower
             - Chord-normalized x-location along the lower surface where transition is forced.
           * - Actuator Disks Active
             - Whether to implement the actuator disk model in the analysis. The remaining fields are disabled if this
               box is not checked.
           * - Num. Actuator Disks
             - The number of actuator disks to use. For the latest version of MSES available at the time of writing
               (3.12), only one actuator disk can be used. Multiple actuator disks require modifications to the source
               code; permission to use modified source code is also required. Email mlauer2@illinois.edu for
               more information about a working version of this modified code.
           * - AD Side
             - Airfoil side from which the actuator disk will emanate. The top of the uppermost airfoil is side 1,
               the bottom of the uppermost airfoil is side 2, the top of the airfoil immediately below the uppermost
               airfoil is side 3, etc.
           * - AD X-Location
             - The :math:`x/c`-location on the "AD Side" from which the actuator disk will emanate. The actuator disk
               starts at this point on the airfoil surface and follows the stream-normal direction of the grid until
               another airfoil surface is reached.
           * - AD X-Location Param
             - Optional parameter allowing the :math:`x/c`-location of the actuator disk to depend on a parameter
               or design variable in the ``GeometryCollection``. The numerical value of this parameter is updated
               automatically during optimization.
           * - AD Total Pres. Ratio
             - The ratio of total pressure immediately downstream to total pressure immediately upstream of the
               actuator disk.
           * - AD Thermal Efficiency
             - The thermal efficiency of the actuator disk.


    .. tab-item:: API
        :sync: api

        *Construction Zone*


MPLOT
-----

MPLOT is the post-processing module in the MSES suite that gives a number of options for exporting data and
generating plots. The *pymead* interface to MPLOT is limited to a small subset of the original features, as
the primary focus is on generating *pymead*-native plots of the flow field and surface data. Of course, MPLOT
can always be used directly from the results in the output folder to generate MSES-native plots.


.. tab-set::

    .. tab-item:: GUI
        :sync: gui

        The MPLOT settings can be accessed from the GUI by navigating in the toolbar to
        **Analysis** |rarrow| **Multi-Element Airfoil** and selecting the "MPLOT" tab.
        Descriptions of the various parameters are given below.

        .. list-table::
           :widths: 20 80
           :header-rows: 1
           :class: max-width-table

           * - Option
             - Description
           * - Timeout (sec)
             - The amount of time allotted to post-processing using MPLOT. The MPLOT process will be automatically
               terminated
               after this amount of time regardless of whether the post-processing has completed.
           * - Output Mach Contours
             - Export the MSES-native Mach contour plot and display it in a new tab on the screen. This option
               also automatically saves the plot as both ``Mach_contours.pdf`` and ``Mach_contours.svg`` in the
               analysis directory. Note that the analysis
               directory can be accessed by clicking on word "MSES" in the console.
           * - Output Streamline Grid
             - Export the streamline grid as ``grid.<airfoil_name>`` to the analysis directory.
           * - Output Grid
             - Export the MSES-native grid plot and display it in a new tab on the screen. This option also
               saves this plot to ``grid.pdf`` and ``grid.svg`` in the analysis directory.
           * - Output Zoomed-In Grid
             - Export a zoomed version of the MSES-native grid plot, display it in a new tab on the screen, and
               save it to both ``grid_zoom.pdf`` and ``grid_zoom.svg``.
           * - Output Flow Variable Field
             - Export the flow variable field data to ``field.<airfoil_name>`` in the analysis directory. This
               step is not performed by default during an MSES runs due the size of the file. While this file is
               usually only 1-2 MB, this size can add up especially during optimization when tens of thousands
               of MSES runs are performed.
           * - Output CPK
             - Compute the mechanical flow power coefficient and store the value in ``aero_data.json`` in the analysis
               directory.


    .. tab-item:: API
        :sync: api

        *Construction Zone*


MPOLAR
------

MPOLAR is the parameter-sweep analysis module in the MSES suite. Currently, the *pymead* interface to MPOLAR is
limited to angle of attack sweeps, which takes advantage of the fact that MSES starts each angle of attack analysis
using the previous angle of attack solution. This means that MPOLAR can run a series of angles of attack much
faster than running MSES individually at each angle of attack.


.. tab-set::

    .. tab-item:: GUI
        :sync: gui

        The MPOLAR settings can be accessed from the GUI by navigating in the toolbar to
        **Analysis** |rarrow| **Multi-Element Airfoil** and selecting the "MPOLAR" tab.
        Descriptions of the various parameters are given below.

        .. list-table::
           :widths: 20 80
           :header-rows: 1
           :class: max-width-table

           * - Option
             - Description
           * - Timeout (s)
             - The amount of time allotted to an MPOLAR analysis. The MPOLAR process will be automatically terminated
               after this amount of time regardless of whether the analysis has completed.
           * - Polar Mode
             - If "No Polar Analysis" is selected, all the settings in the MPOLAR tab will be ignored. If
               "Alpha Sweep from Data File" is selected, the proceeding field ("Angle of Attack") will be used
               to determine the array of angles of attack to analyze from a file. If "Alpha Sweep from Start/Stop/Inc"
               is selected, the "α Sweep Start (deg)," "α Sweep End (deg)," and "α Increment (deg)" fields will be
               used to compute the array of angles of attack.
           * - α Sweep Start (deg)
             - Starting value for the angle of attack sweep in degrees. Ignored if "Alpha Sweep from Data File" is
               selected.
           * - α Sweep End (deg)
             - Ending value for the angle of attack sweep in degrees. Ignored if "Alpha Sweep from Data File" is
               selected.
           * - α Increment (deg)
             - Increment for the angle of attack sweep in degrees. Ignored if "Alpha Sweep from Data File" is
               selected. Note that MPOLAR will automatically decrease the increment at higher angles of attack
               if the prescribed angle of attack cannot be converged.



    .. tab-item:: API
        :sync: api

        *Construction Zone*


References
==========

[1] J. Katz and A. Plotkin, Low-Speed Aerodynamics, Second Edition, 2nd ed. New York, NY,
    USA: Cambridge University Press, 2004. Accessed: Mar. 07, 2023. [Online].
    Available: `<https://asmedigitalcollection.asme.org/fluidsengineering/article/126/2/293/458666/LowSpeed-Aerodynamics-Second-Edition>`_
