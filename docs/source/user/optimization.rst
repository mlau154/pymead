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
             - The aerodynamic functions to be minimized (separated by commas if there are more than one).

    .. tab-item:: Constraints/Termination

        **Constraints/Termination**

        Settings for a variety of geometric constraints and algorithm termination settings

    .. tab-item:: Multi-Point Optimization

        **Multi-Point Optimization**

        Setup for a multi-point stencil
