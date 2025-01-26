Optimization
############

``pymead`` has the capability of performing aerodynamic and aeropropulsive shape optimization on airfoil systems
using a genetic algorithm (GA). Currently, the only algorithm available is the robust
U-NSGA-III algorithm (see `here <https://pymoo.org/algorithms/moo/unsga3.html>`_ for details).
Any of the values available in the aerodynamic data output from the flow solver may be used in one ore more
mathematical expressions to define the objective function(s). The multi-point optimization feature allows for an
increase in robustness of design by considering one or more off-design conditions for the airfoil(s).

The following is a walk-through of each of the optimization-specific tabs in the *Optimization Setup* dialog.
Information about the **XFOIL**, **MSET**, **MSES**, and **MPLOT** tabs can be found in :ref:`aero-analysis`.

.. tab-set::

    .. tab-item:: General Settings

        **General Settings**

        General settings for the GA problem.

        .. list-table::
           :widths: 20 80
           :header-rows: 1
           :class: max-width-table

           * - Option
             - Description
           * - Save
             - Save the current optimization settings to the currently cached save file
           * - Save Settings As...
             - Save the current optimization settings to a new ``.json`` file
           * - Load
             - Load optimization settings from a ``.json`` file
           * - Warm Start Active?
             - Whether to start from a previously terminated optimization. If this box is checked, the
               *Warm Start Generation* and *Warm Start Directory* must both be specified
           * - Warm Start Generation
             - Which generation to start from. Use ``-1`` to start from the most recently completed generation.
           * - Warm Start Directory
             - The directory from which to load in the data for the partially completed optimization. This folder
               should contain a number of files of the form ``algorithm_gen_XXX.pkl`` and ``opt_gen_XXX.jmea``.
           * - Use Initial Settings?
             - Whether to use the initial set of optimizations on warm start. Normally this box should be checked.
               Unchecking this box allows for a different number of offspring to be applied from the currently
               displayed optimization settings dialog to the warm start optimization. No other changes to settings
               will be applied.
           * - MEA File
             - The geometry collection (``.jmea`` file) to use as a baseline for the optimization. Note that the
               file chosen here will be loaded into the GUI. While running an optimization with this option empty
               will correctly use the last saved state of the ``.jmea`` file currently loaded into the GUI, it is
               normally recommended to fill in this line so that a reference to the ``.jmea`` file used will be stored
               in the saved optimization settings.
           * - Batch Mode Active?
             - Whether to run the list of optimization files provided in the "Batch Settings Files" option
           * - Batch Settings Files
             - Runs a list of ``.json`` optimization settings files. This option is experimental and is not
               guaranteed to run correctly.

    .. tab-item:: Genetic Algorithm

        **Genetic Algorithm**

        Parameters specific to the ``pymoo`` implementation of the genetic algorithm

        .. list-table::
           :widths: 20 80
           :header-rows: 1
           :class: max-width-table

           * - Option
             - Description
           * - CFD Tool
             - The analysis tool to be used during the optimization run
           * - Objective Functions
             - The aerodynamic functions to be minimized (separated by commas if there are more than one). Available
               functions depend on the CFD tool selected and are referenced with a leading dollar sign ($). See
               :ref:`obj-functions` for a list of available functions. Note that the functions must be indexed
               if multipoint optimization is active. For example, for an even weighting of drag coefficient
               across two multipoint stencil points, use ``0.5 * $Cd[0] + 0.5 * $Cd[1]``. Note that
               ``$Cd[0] + $Cd[1]$`` is equivalent from an optimization perspective, but requiring that the weights
               sum to one allows the objective function to more intuitively represent the average drag coefficient
               across stencil points.
           * - Aerodynamic Constraints
             - Similar to "Objective Functions." Negative constraint values are feasible. As an example, use
               ``-($Cm + 0.1)`` to enforce :math:`C_m \geq -0.1` for a single-point optimization, or
               ``-($Cm[0] + 0.10),-($Cm[1] + 0.12)`` to enforce :math:`C_m \geq -0.10` and :math:`C_m \geq -0.12` for the
               first two points in a multipoint optimization, respectively. To enforce a lift coefficient constraint,
               it is recommended to implicitly enforce this constraint by setting the lift coefficient value instead
               of angle of attack in the XFOIL or MSES settings tab, as this generally gives better results since it
               automatically enforces the lift coefficient value during the analysis.
           * - Population Size
             - Number of airfoils or airfoil systems required to be converged during each generation. This number
               should generally be increased with increasing number of design variables
           * - Number of Offspring
             - The maximum number of airfoils or airfoil systems allowed to be analyzed during each generation. If this
               number is reached during a given generation, the optimization will terminate due to too many
               infeasible solutions. This number should generally be much larger than the population size to allow
               for some self-intersecting geometries or unconverged XFOIL/MSES solutions to occur before determining
               that the optimization should be terminated. Because the airfoil systems are generated lazily
               in the parallel-processing scheme (i.e., not generated before each generation begins), setting this
               value to a large number does not incur a significant computational time penalty.
           * - Max. Sampling Width
             - The amount in bounds-normalized space that each design variable is allowed to be perturbed during
               the sampling stage. For example, a design variable with bounds :math:`[2.0,3.0]` and an initial value
               of :math:`2.3` can be set to any value in the range :math:`[2.2,2.4]` if the maximum sampling width
               is set to :math:`0.1`. This is done to avoid producing too many "poor" geometries during the sampling
               stage, especially if the design variables are given wide bounds. For design variable sets with tight
               bounds, this value should be set to a larger value (e.g., ``0.3`` or ``0.4``) to allow maximum
               diversity in the sampling generation.
           * - :math:`\eta` (crossover)
             - Crossover constant determining the width of the sampling distribution when performing the crossover
               step for each generation. Larger values allow for larger perturbations in design variable values.
               See `this page <https://pymoo.org/operators/crossover.html#Simulated-Binary-Crossover-(SBX)>`_ for more
               details.
           * - :math:`\eta` (mutation)
             - Mutation constant determining the width of the sampling distribution when performing the random mutation
               step for each generation. Larger values allow for larger perturbations in design variable values.
               See `this page <https://pymoo.org/operators/mutation.html#Polynomial-Mutation-(PM)>`_ for more
               details.
           * - Random Seed
             - Psuedo-random number generator seed to use such that the same results are obtained every time the
               optimization is run with identical settings.
           * - Number of Processors
             - Number of logical threads to be used for analyzing airfoil geometries in parallel. The maximum
               number is that given by ``os.cpu_count()``. Note that for a CPU with six dual-threaded cores,
               this value can be set up to ``12``.
           * - State Save Frequency
             - How often to save the state of the optimization. If ``1``, the optimization data will be saved
               after every generation.
           * - Opt. Root Directory
             - The base location for the optimization folders. Most information will be stored in
               ``<Opt. Root Directory>/<Opt. Directory Name>``.
           * - Opt. Directory Name
             - Sub-directory where the optimization data will be stored
           * - Temp. Analysis Dir. Name
             - Name of the base directory for all XFOIL/MSES analysis data which gets overridden every generation


    .. tab-item:: Constraints/Termination

        **Constraints/Termination**

        Settings for a variety of geometric constraints and algorithm termination settings

    .. tab-item:: Multi-Point Optimization

        **Multi-Point Optimization**

        Setup for a multi-point stencil


.. _obj-functions:

Available Objective/Constraint Functions
========================================

XFOIL
-----
- ``$Cd`` (drag coefficient)
- ``$Cl`` (lift coefficient)
- ``$alf`` (angle of attack, degrees)
- ``$Cm`` (pitching moment coefficient)
- ``$Cdf`` (friction drag coefficient)
- ``$Cdp`` (pressure drag coefficient)

MSES
----
Same as XFOIL, with the addition of

- ``$Cdv`` (viscous drag coefficient, equivalent to ``$Cd - $Cdw - $Cdh``)
- ``$Cdw`` (wave drag coefficient)
- ``$Cdh`` (actuator disk "drag" coefficient; generally negative if the actuator disk fan pressure ratio
  is greater than ``1.0``)
- ``$CPK`` (mechanical flow power coefficient; only available if the "Output CPK" option is selected in
  the MPLOT settings)

