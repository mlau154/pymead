import os
import subprocess
import time
import typing
from collections import namedtuple
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from shapely.geometry import MultiPoint
import multiprocessing.connection

from pymead.analysis.compressible_flow import calculate_normal_shock_total_pressure_ratio
from pymead.analysis.read_aero_data import read_aero_data_from_xfoil, read_Cp_from_file_xfoil, read_bl_data_from_mses, \
    read_forces_from_mses, read_grid_stats_from_mses, read_field_from_mses, read_streamline_grid_from_mses, \
    flow_var_idx, convert_blade_file_to_3d_array, read_actuator_disk_data_mses, read_Mach_from_mses_file
from pymead.core.mea import MEA
from pymead.utils.file_conversion import convert_ps_to_svg
from pymead.utils.geometry import convert_numpy_array_to_shapely_LineString
from pymead.utils.read_write_files import save_data

SVG_PLOTS = ['Mach_contours', 'grid', 'grid_zoom']
SVG_SETTINGS_TR = {
    SVG_PLOTS[0]: 'Mach',
    SVG_PLOTS[1]: 'Grid',
    SVG_PLOTS[2]: 'Grid_Zoom',
}


class XFOILSettings:

    mode_prescribe_mapping = {0: "Angle of Attack (deg)", 1: "Viscous Cl", 2: "Inviscid Cl"}

    def __init__(self,
                 base_dir: str,
                 airfoil_name: str,
                 Re: float = 1.0e6,
                 Ma: float = 0.1,
                 mode: int = 0,
                 timeout: float = 8.0,
                 iterations: int = 150,
                 xtr: typing.List[float] = None,
                 N: float = 9.0,
                 visc: bool = True,
                 alfa: float = 0.0,
                 Cl: float = 0.0,
                 Cli: float = 0.0
                 ):
        """
        Defines a set of reasonable default inputs to ``run_xfoil`` or ``calculate_aero_data`` with
        ``tool="XFOIL"``.

        Parameters
        ----------
        base_dir: str
            Base directory for the airfoil analysis. All the files used in an XFOIL analysis
            can be found in ``base_dir/airfoil_name``

        airfoil_name: str
            Used to name the end of the file path where the airfoil analysis will take place

        Re: float
            Reynolds number (ignored if ``visc==False``). Default: ``1.0e6``

        Ma: float
            Mach number. Values near or above ``1.0`` will produce highly inaccurate results due to
            unmodeled shock waves. Default: ``0.1``

        mode: int
            If ``0``, the angle of attack defined by ``alfa`` is prescribed.
            If ``1``, the viscous lift coefficient defined by ``Cl`` is prescribed.
            If ``2``, the inviscid lift coefficient defined by ``Cli`` is prescribed.

        timeout: float
            If XFOIL runs longer than this time (in seconds), the run will
            be terminated prior to convergence. Default: ``8.0``

        iterations: int
            Maximum number of iterations allowed by XFOIL in viscous mode. Default: ``150``

        xtr: typing.List[float] or None
            Two-element list consisting of :math:`x/c`-locations of the upper and lower airfoil surfaces. If
            ``None`` is specified, a default value of ``[1.0, 1.0]`` will be used. This value corresponds
            to free transition on both surfaces.

        N: float
            Envelope method exponent, 9.0 for an average wind tunnel. See the "Transition Criterion" section of the
            `XFOIL user guide <https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt>`_
            for more details and additional flow conditions. Default: ``9.0``

        visc: bool
            Whether to include a boundary layer model in the airfoil analysis. Default: ``True``

        alfa: float
            Angle of attack in degrees. Ignored unless ``mode==0``. Default: ``0.0``

        Cl: float
            Viscous lift coefficient to prescribe. Ignored unless ``mode==1``. Default: ``0.0``

        Cli: float
            Inviscid lift coefficient to prescribe. Ignored unless ``mode==2``. Default: ``0.0``
        """
        self.Re = Re
        self.Ma = Ma
        self.mode = mode
        self.prescribe = self.mode_prescribe_mapping[self.mode]
        self.timeout = timeout
        self.iterations = iterations
        self.xtr = [1.0, 1.0] if xtr is None else xtr
        self.N = N
        self.visc = visc
        self.base_dir = base_dir
        self.airfoil_name = airfoil_name
        self.alfa = alfa
        self.Cl = Cl
        self.Cli = Cli
        if len(self.xtr) != 2:
            raise ValueError("'xtr' must be a list containing exactly two values: the x/c transition location for the "
                             "upper surface and the x/c transition location for the lower surface")
        if self.mode not in self.mode_prescribe_mapping.keys():
            raise ValueError(f"'mode' must be an integer (either 0, 1, or 2) corresponding to the following prescribed "
                             f"modes for XFOIL: {self.mode_prescribe_mapping}")

    def get_dict_rep(self) -> dict:
        """
        Gets a Python dictionary description of the XFOIL analysis parameters. Used in ``run_xfoil`` and
        ``calculate_aero_data``.

        Returns
        -------
        dict
            XFOIL analysis parameters
        """
        return {
            "Re": self.Re,
            "Ma": self.Ma,
            "prescribe": self.prescribe,
            "timeout": self.timeout,
            "iter": self.iterations,
            "xtr": self.xtr,
            "N": self.N,
            "base_dir": self.base_dir,
            "airfoil_name": self.airfoil_name,
            "visc": self.visc,
            "alfa": self.alfa,
            "Cl": self.Cl,
            "CLI": self.Cli
        }


class MSETSettings:
    def __init__(self,
                 mea: str,
                 grid_bounds: typing.List[float] = None,
                 airfoil_side_points: int = 180,
                 exp_side_points: float = 0.9,
                 inlet_pts_left_stream: int = 41,
                 outlet_pts_right_stream: int = 41,
                 num_streams_top: int = 17,
                 num_streams_bot: int = 23,
                 max_streams_between: int = 15,
                 elliptic_param: float = 1.3,
                 stag_pt_aspect_ratio: float = 2.5,
                 x_spacing_param: float = 0.85,
                 alf0_stream_gen: float = 0.0,
                 timeout: float = 10.0
                 ):
        self.grid_bounds = grid_bounds if grid_bounds is not None else [-5.0, 5.0, -5.0, 5.0]


class MSESSettings:
    def __init__(self):
        pass


class MPLOTSettings:
    def __init__(self):
        pass


def update_xfoil_settings_from_stencil(xfoil_settings: dict, stencil: typing.List[dict], idx: int):
    """
    Updates the XFOIL settings dictionary from a given multipoint stencil and multipoint index

    Parameters
    ==========
    xfoil_settings: dict
      MSES settings dictionary

    stencil: typing.List[dict]
      A list of dictionaries describing the multipoint stencil, where each entry in the list is a dictionary
      representing a different stencil variable (Mach number, lift coefficient, etc.) and contains values for the
      variable name, index (not used in XFOIL), and stencil
      point values (e.g., ``stencil_var["points"]`` may look something like ``[0.65, 0.70, 0.75]`` for Mach number)

    idx: int
      Index within the multipoint stencil used to update the XFOIL settings dictionary

    Returns
    =======
    dict
      The modified XFOIL settings dictionary
    """
    for stencil_var in stencil:
        if isinstance(xfoil_settings[stencil_var['variable']], list):
            xfoil_settings[stencil_var['variable']][stencil_var['index']] = stencil_var['points'][idx]
        else:
            xfoil_settings[stencil_var['variable']] = stencil_var['points'][idx]
    return xfoil_settings


def update_mses_settings_from_stencil(mses_settings: dict, stencil: typing.List[dict], idx: int):
    """
    Updates the MSES settings dictionary from a given multipoint stencil and multipoint index

    Parameters
    ==========
    mses_settings: dict
      MSES settings dictionary

    stencil: typing.List[dict]
      A list of dictionaries describing the multipoint stencil, where each entry in the list is a dictionary
      representing a different stencil variable (Mach number, lift coefficient, etc.) and contains values for the
      variable name, index (used only in the case of transition location and actuator disk variables), and stencil
      point values (e.g., ``stencil_var["points"]`` may look something like ``[0.65, 0.70, 0.75]`` for Mach number)

    idx: int
      Index within the multipoint stencil used to update the MSES settings dictionary

    Returns
    =======
    dict
      The modified MSES settings dictionary
    """
    for stencil_var in stencil:
        if isinstance(mses_settings[stencil_var['variable']], list):
            mses_settings[stencil_var['variable']][stencil_var['index']] = stencil_var['points'][idx]
        else:
            mses_settings[stencil_var['variable']] = stencil_var['points'][idx]
    return mses_settings


def calculate_aero_data(conn: multiprocessing.connection.Connection or None,
                        airfoil_coord_dir: str,
                        airfoil_name: str,
                        coords: typing.List[np.ndarray] or np.ndarray = None,
                        mea: MEA = None,
                        mea_airfoil_names: typing.List[str] = None,
                        tool: str = 'XFOIL',
                        xfoil_settings: dict = None,
                        mset_settings: dict = None,
                        mses_settings: dict = None,
                        mplot_settings: dict = None,
                        export_Cp: bool = True,
                        save_aero_data: bool = True):
    r"""
    Convenience function calling either XFOIL or MSES depending on the ``tool`` specified

    Parameters
    ==========
    conn: multiprocessing.connection.Connection or None
        If not ``None``, a connection established between the worker thread deployed by the GUI over which
        data can be passed to update the user on the state of the analysis

    airfoil_coord_dir: str
        The directory containing the airfoil coordinate file

    airfoil_name: str
        A string describing the airfoil

    coords: typing.Tuple[tuple]
        If using XFOIL: specify a 2-D nested tuple array of size :math:`N \times 2`, where :math:`N` is the number of
        airfoil coordinates and the columns represent :math:`x` and :math:`y`. If using MSES, specify a 3-D nested
        tuple array of size :math:`M \times N \times 2`, where :math:`M` is the number of airfoils.

    mea: MEA or None
        Multi-element airfoil object to use if ``coords`` is not specified. Default: ``None``

    mea_airfoil_names: typing.List[str] or None
        Names of the airfoils contained in the ``mea`` to analyze

    tool: str
        The airfoil flow analysis tool to be used. Must be either ``"XFOIL"`` or ``"MSES"``. Default: ``"XFOIL"``

    xfoil_settings: dict
        A dictionary containing the settings for XFOIL. Must be specified if the ``"XFOIL"`` tool is selected.
        Default: ``None``

    mset_settings: dict
      A dictionary containing the settings for MSET. Must be specified if the ``"MSES"`` tool is selected. Default:
      ``None``

    mses_settings: dict
      A dictionary containing the settings for MSES. Must be specified if the ``"MSES"`` tool is selected. Default:
      ``None``

    mplot_settings: dict
      A dictionary containing the settings for MPLOT. Must be specified if the ``"MSES"`` tool is selected. Default:
      ``None``

    export_Cp: bool
      Whether to calculate and export the surface pressure coefficient distribution in the case of XFOIL, or the
      entire set of boundary layer data in the case of MSES. Default: ``True``

    save_aero_data: bool
        Whether to save the aerodynamic data as a JSON file to the analysis directory

    Returns
    =======
    dict, str
        A dictionary containing the evaluated aerodynamic data and the path to the log file
    """

    def send_over_pipe(data: object):
        """
        Connection to the GUI that is only used if ``calculate_aero_data`` is being called directly from the GUI

        Parameters
        ----------
        data: object
            The intermediate information to pass to the GUI, normally a two-element tuple where the first argument
            is a string specifying the kind of data being sent, and the second argument being the actual data
            itself (note that the data must be picklable by the multiprocessing module)

        Returns
        -------

        """
        try:
            if conn is not None:
                conn.send(data)
        except BrokenPipeError:
            pass

    tool_list = ['XFOIL', 'MSES']
    if tool not in tool_list:
        raise ValueError(f"\'tool\' must be one of {tool_list}")
    # if tool == 'XFOIL':
    #
    #     # Check for self-intersection and early return if self-intersecting:
    #     if check_airfoil_self_intersection(coords):
    #         return False

    aero_data = {}

    # Make the analysis directory if not already created
    base_dir = os.path.join(airfoil_coord_dir, airfoil_name)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    if tool == 'XFOIL':
        if xfoil_settings is None:
            raise ValueError(f"\'xfoil_settings\' must be set if \'xfoil\' tool is selected")

        xfoil_log = None

        # Set up single-point or multipoint settings
        xfoil_loop_iterations = 1
        stencil = None
        aero_data_list = None
        if 'multi_point_stencil' in xfoil_settings.keys() and xfoil_settings['multi_point_stencil'] is not None:
            stencil = xfoil_settings['multi_point_stencil']
            xfoil_loop_iterations = len(stencil[0]['points'])
            aero_data_list = []

        # Multipoint Loop
        for i in range(xfoil_loop_iterations):

            if stencil is not None:
                xfoil_settings = update_xfoil_settings_from_stencil(xfoil_settings=xfoil_settings, stencil=stencil, idx=i)
                # print(f"{mses_settings['XCDELH'] = }, {mses_settings['CLIFIN'] = }, {mses_settings['PTRHIN'] = }")

            aero_data, xfoil_log = run_xfoil(xfoil_settings, coords, export_Cp=export_Cp)

            if aero_data["converged"]:
                if aero_data_list is not None:
                    aero_data_list.append(aero_data)
            else:
                if aero_data_list is not None:
                    aero_data_list.append(aero_data)
                break

        if aero_data_list is not None:
            aero_data = {k: [] for k in aero_data_list[0].keys()}
            for aero_data_set in aero_data_list:
                for k, v in aero_data_set.items():
                    aero_data[k].append(v)

        if save_aero_data:
            if aero_data["converged"]:
                for k, v in aero_data["Cp"].items():
                    if isinstance(v, np.ndarray):
                        aero_data["Cp"][k] = v.tolist()
            save_data(aero_data, os.path.join(base_dir, "aero_data.json"))

        return aero_data, xfoil_log

    elif tool in ['mses', 'Mses', 'MSES']:
        aero_data['converged'] = False
        aero_data['timed_out'] = False
        aero_data['errored_out'] = False
        if mset_settings is None:
            raise ValueError(f"\'mset_settings\' must be set if \'mses\' tool is selected")
        if mses_settings is None:
            raise ValueError(f"\'mses_settings\' must be set if \'mses\' tool is selected")
        if mplot_settings is None:
            raise ValueError(f"\'mplot_settings\' must be set if \'mses\' tool is selected")

        converged = False
        mses_log, mplot_log = None, None
        mset_success, mset_log, airfoil_name_order = run_mset(
            airfoil_name, airfoil_coord_dir, mset_settings, coords=coords, mea=mea, mea_airfoil_names=mea_airfoil_names)

        send_over_pipe(("message", f"MSET success"))

        # Set up single-point or multipoint settings
        mset_mplot_loop_iterations = 1
        stencil = None
        aero_data_list = None
        if 'multi_point_stencil' in mses_settings.keys() and mses_settings['multi_point_stencil'] is not None:
            stencil = mses_settings['multi_point_stencil']
            mset_mplot_loop_iterations = len(stencil[0]['points'])
            aero_data_list = []

        # Multipoint Loop
        for i in range(mset_mplot_loop_iterations):
            t1 = time.time()

            if stencil is not None:
                mses_settings = update_mses_settings_from_stencil(mses_settings=mses_settings, stencil=stencil, idx=i)
                # print(f"{mses_settings['XCDELH'] = }, {mses_settings['CLIFIN'] = }, {mses_settings['PTRHIN'] = }")

            if mset_success:
                t_start = time.perf_counter()
                converged, mses_log = run_mses(airfoil_name, airfoil_coord_dir, mses_settings,
                                               airfoil_name_order=airfoil_name_order, conn=conn)
                t_end = time.perf_counter()
                send_over_pipe(("message", f"MSES converged in {t_end-t_start:.2f} seconds"))
            if mset_success and converged:
                mplot_log = run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode='forces')
                aero_data = read_forces_from_mses(mplot_log)

                # This is error is triggered in read_aero_data() in the rare case that there is an error reading the
                # force file. If this error is triggered, break out of them multipoint loop.
                errored_out = np.isclose(aero_data["Cd"], 1000.0)
                if errored_out:
                    aero_data["converged"] = True
                    aero_data["timed_out"] = False
                    aero_data['errored_out'] = True
                    if aero_data_list is not None:
                        aero_data_list.append(aero_data)
                    break

                if export_Cp:
                    run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode="Cp")
                    aero_data["BL"] = []
                    for attempt in range(100):
                        if attempt > 0:
                            print(f"{attempt = }")
                        try:
                            bl = read_bl_data_from_mses(os.path.join(airfoil_coord_dir, airfoil_name,
                                                                     f"bl.{airfoil_name}"))
                            for side in range(len(bl)):
                                aero_data["BL"].append({})
                                for output_var in ["x", "y", "Cp"]:
                                    aero_data["BL"][-1][output_var] = bl[side][output_var]
                            break
                        except KeyError:
                            time.sleep(0.01)

                if mplot_settings["CPK"]:
                    mplot_settings["flow_field"] = 2
                    mplot_settings["Streamline_Grid"] = 2

                if mplot_settings["flow_field"]:
                    mplot_settings["Streamline_Grid"] = 2

                for mplot_output_name in ['Mach', 'Streamline_Grid', 'Grid', 'Grid_Zoom', 'flow_field']:
                    if mplot_settings[mplot_output_name]:
                        run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode=mplot_output_name)
                        if mplot_output_name == 'flow_field':
                            run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode='grid_stats')

                if mplot_settings["CPK"]:
                    try:
                        # outputs_CPK = calculate_CPK_power_consumption(os.path.join(airfoil_coord_dir, airfoil_name))
                        # outputs_CPK = calculate_CPK_power_consumption_old(os.path.join(airfoil_coord_dir, airfoil_name))
                        outputs_CPK = calculate_CPK_mses_inviscid_only(os.path.join(airfoil_coord_dir, airfoil_name))
                        send_over_pipe(("message", "CPK calculation success"))
                    except Exception as e:
                        print(f"{e = }")
                        send_over_pipe(("message", "CPK calculation failed"))
                        # outputs_CPK = {"CPK": 1e9, "diss_shock": 1e9, "diss_surf": 1e9, "Edot": 1e9, "Cd": 1e9,
                        #                "Edota": 1e9, "Edotp": 1e9}
                        outputs_CPK = {"CPK": 1e9}
                    aero_data = {**aero_data, **outputs_CPK}

            t2 = time.time()
            # print(f"Time for stencil point {i}: {t2 - t1} seconds")

            if converged:
                aero_data['converged'] = True
                aero_data['timed_out'] = False
                aero_data['errored_out'] = False
                if aero_data_list is not None:
                    aero_data_list.append(aero_data)
            else:
                aero_data['converged'] = False
                aero_data["timed_out"] = False
                aero_data["errored_out"] = False
                if aero_data_list is not None:
                    aero_data_list.append(aero_data)
                break

            # print(f"{aero_data = }")

        logs = {'mset': mset_log, 'mses': mses_log, 'mplot': mplot_log}

        if aero_data_list is not None:
            aero_data = {k: [] for k in aero_data_list[0].keys()}
            for aero_data_set in aero_data_list:
                for k, v in aero_data_set.items():
                    aero_data[k].append(v)

        send_over_pipe(
            ("mses_analysis_complete", (aero_data, mset_settings, mses_settings, mplot_settings, mea.name()))
        )

        if save_aero_data:
            save_data(aero_data, os.path.join(airfoil_coord_dir, airfoil_name, "aero_data.json"))

        send_over_pipe(("message", "Post-processing successful"))

        return aero_data, logs


def calculate_Cl_integral_form(x: np.ndarray, y: np.ndarray, Cp: np.ndarray, alfa: float):
    # Calculate the lift coefficient (integral from 0 to max arc length of Cp*(nhat dot jhat)*dl)
    panel_length = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])  # length of the panels
    panel_nhat_angle = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1]) + np.pi / 2  # angle of the panel outward normal vector
    panel_center_x = 0.5 * (x[1:] + x[:-1])
    panel_center_y = 0.5 * (y[1:] + y[:-1])
    quarter_chord_center_x = 0.25
    quarter_chord_center_y = 0.0
    panel_nhat_jcomp = np.sin(panel_nhat_angle - alfa)  # j-component of the panel normal vector
    # panel_nhat_icomp = np.cos(panel_nhat_angle - alfa)
    panel_Cp = (Cp[1:] + Cp[:-1]) / 2  # average pressure coefficient of the panel
    panel_force = panel_Cp * np.array([np.cos(panel_nhat_angle - np.pi),
                                       np.sin(panel_nhat_angle - np.pi)]) * panel_length
    r_vec = np.array([panel_center_x - quarter_chord_center_x, panel_center_y - quarter_chord_center_y])
    panel_Cm = np.cross(r_vec.T, panel_force.T)
    Cm = np.sum(panel_Cm)
    Cl = np.sum(panel_Cp * panel_nhat_jcomp * panel_length)
    # Cdp = np.sum(panel_Cp * panel_nhat_icomp * panel_length)
    return Cl, Cm


def read_alfa_from_xfoil_cp_file(xfoil_cp_file: str):
    with open(xfoil_cp_file, "r") as f:
        lines = f.readlines()
    return float(lines[1].split()[2])


def calculate_Cl_alfa_xfoil_inviscid(airfoil_name: str, base_dir: str):
    cp_file = os.path.join(base_dir, f"{airfoil_name}_Cp.dat")
    alfa = read_alfa_from_xfoil_cp_file(cp_file)
    data = np.loadtxt(cp_file, skiprows=3)
    x, y, Cp = data[:, 0], data[:, 1], data[:, 2]
    Cl, Cm = calculate_Cl_integral_form(x, y, Cp, np.deg2rad(alfa))
    return Cl, Cm, alfa


def run_xfoil(xfoil_settings: dict or XFOILSettings, coords: np.ndarray, export_Cp: bool = True) -> (dict, str):
    """
    Python wrapper for `XFOIL <https://web.mit.edu/drela/Public/web/xfoil/>`_

    Parameters
    ----------
    xfoil_settings: dict or XFOILSettings
        Analysis parameters for XFOIL. If a ``dict`` is used, the keys found in ``XFOILSettings.get_dict_rep``
        must be specified

    coords: numpy.ndarray
        Airfoil coordinates in Selig format (counter-clockwise starting from the trailing edge upper surface).

    export_Cp: bool
        Whether to export the surface pressure coefficient distribution. Default: ``True``

    Returns
    -------
    dict, str
        A dictionary containing the aerodynamic performance data and the path to the XFOIL log file.
        The dictionary will have at least the following keys: ``"converged"``, ``"timed_out"``, ``"errored_out"``,
        ``"Cl"`` (lift coefficient), ``"Cm"`` (pitching moment coefficient), and ``"alf"`` (angle of attack in degrees).
        If run in viscous mode (``visc==True``), the following keys will also be included: ``"Cd"`` (drag coefficient),
        ``"Cdf"`` (friction drag coefficient), ``"Cdp"`` (pressure drag coefficient), and ``"L/D"``
        (lift-to-drag ratio). If ``export_Cp==True``, the key ``"Cp"`` (surface pressure coefficient distribution)
        will also be included. The :math:`C_p` distribution is a dictionary containing the following keys, where each
        of the values is a one-dimensional ``numpy.ndarray``: ``"x"``, ``"y"``, and ``"Cp"``.
    """

    aero_data = {}

    if isinstance(xfoil_settings, XFOILSettings):
        xfoil_settings = xfoil_settings.get_dict_rep()

    airfoil_name = xfoil_settings["airfoil_name"]
    base_dir = xfoil_settings["base_dir"]
    analysis_dir = os.path.join(base_dir, airfoil_name)

    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    if "xtr" not in xfoil_settings.keys():
        xfoil_settings["xtr"] = [1.0, 1.0]
    if 'N' not in xfoil_settings.keys():
        xfoil_settings['N'] = 9.0
    f = os.path.join(analysis_dir, airfoil_name + ".dat")

    # Attempt to save the file
    save_attempts = 0
    max_save_attempts = 100
    while True:
        save_attempts += 1
        if save_attempts > max_save_attempts:
            raise ValueError("Exceeded the maximum number of allowed coordinate file save attempts")
        try:
            np.savetxt(f, coords)
            break
        except OSError:
            time.sleep(0.01)

    xfoil_input_file = os.path.join(analysis_dir, 'xfoil_input.txt')

    if xfoil_settings["visc"]:
        xfoil_input_list = ['', 'oper', f'iter {xfoil_settings["iter"]}', 'visc', str(xfoil_settings['Re']),
                            f'M {xfoil_settings["Ma"]}',
                            'vpar', f'xtr {xfoil_settings["xtr"][0]} {xfoil_settings["xtr"][1]}',
                            f'N {xfoil_settings["N"]}', '']
    else:
        xfoil_input_list = ["", "oper", f"iter {xfoil_settings['iter']}", f"M {xfoil_settings['Ma']}"]

    # alpha/Cl input setup (must choose exactly one of alpha, Cl, or CLI)
    alpha = None
    Cl = None
    CLI = None
    if xfoil_settings["prescribe"] == "Angle of Attack (deg)":
        alpha = xfoil_settings["alfa"]
    elif xfoil_settings["prescribe"] == "Viscous Cl":
        Cl = xfoil_settings["Cl"]
    elif xfoil_settings["prescribe"] == "Inviscid Cl":
        CLI = xfoil_settings["CLI"]
    if alpha is not None:
        if not isinstance(alpha, list):
            alpha = [alpha]
        for idx, alf in enumerate(alpha):
            xfoil_input_list.append(f"alfa {alf}")
    elif Cl is not None:
        if not isinstance(Cl, list):
            Cl = [Cl]
        for idx, Cl_ in enumerate(Cl):
            xfoil_input_list.append(f"Cl {Cl_}")
    elif CLI is not None:
        if not isinstance(CLI, list):
            CLI = [CLI]
        for idx, CLI_ in enumerate(CLI):
            xfoil_input_list.append(f"CLI {CLI_}")
    else:
        raise ValueError('At least one of alpha, Cl, or CLI must be set for XFOIL analysis.')

    if export_Cp:
        xfoil_input_list.append('cpwr ' + f"{airfoil_name}_Cp.dat")
    xfoil_input_list.append('')
    xfoil_input_list.append('quit')
    write_input_file(xfoil_input_file, xfoil_input_list)
    xfoil_log = os.path.join(analysis_dir, 'xfoil.log')
    with open(xfoil_input_file, 'r') as g:
        process = subprocess.Popen(['xfoil', f"{airfoil_name}.dat"], stdin=g, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd=analysis_dir, shell=False)
        aero_data['converged'] = False
        aero_data['timed_out'] = False
        aero_data['errored_out'] = False
        try:
            outs, errs = process.communicate(timeout=xfoil_settings['timeout'])
            with open(xfoil_log, 'wb') as h:
                h.write('Output:\n'.encode('utf-8'))
                h.write(outs)
                h.write('\nErrors:\n'.encode('utf-8'))
                h.write(errs)
            aero_data['timed_out'] = False
            aero_data['converged'] = True

            # This commented code is currently not specific enough to handle only global convergence failure
            # (and not catch local convergence failures)
            # with open(xfoil_log, "r") as log_file:
            #     for line in log_file:
            #         if "Convergence failed" in line:
            #             print(f"Convergence failed! {log_file = }")
            #             aero_data["converged"] = False
            #             break

        except subprocess.TimeoutExpired:
            process.kill()
            outs, errs = process.communicate()
            with open(xfoil_log, 'wb') as h:
                h.write('After timeout, \nOutput: \n'.encode('utf-8'))
                h.write(outs)
                h.write('\nErrors:\n'.encode('utf-8'))
                h.write(errs)
            aero_data['timed_out'] = True
            aero_data['converged'] = False
        finally:
            if xfoil_settings["visc"]:
                if not aero_data['timed_out'] and aero_data["converged"]:
                    line1, line2 = read_aero_data_from_xfoil(xfoil_log, aero_data)
                    if line1 is not None:
                        convert_xfoil_string_to_aero_data(line1, line2, aero_data)
                        if export_Cp:
                            aero_data['Cp'] = read_Cp_from_file_xfoil(
                                os.path.join(analysis_dir, f"{airfoil_name}_Cp.dat")
                            )
            else:
                if not aero_data["timed_out"] and aero_data["converged"]:
                    aero_data["Cl"], aero_data["Cm"], aero_data["alf"] = calculate_Cl_alfa_xfoil_inviscid(
                        airfoil_name=airfoil_name, base_dir=analysis_dir)
                    if export_Cp:
                        aero_data["Cp"] = read_Cp_from_file_xfoil(
                            os.path.join(analysis_dir, f"{airfoil_name}_Cp.dat")
                        )

    return aero_data, xfoil_log


def run_mset(name: str, base_dir: str, mset_settings: dict, mea_airfoil_names: typing.List[str],
             coords: typing.List[np.ndarray] = None, mea: MEA = None) -> (bool, str, typing.List[str]):
    r"""
    A Python wrapper for MSET

    Parameters
    ----------
    name: str
        Name of the airfoil [system]

    base_dir: str
        MSET files will be stored in ``base_dir/name``

    mset_settings: dict
        Analysis parameter set (dictionary)

    coords: typing.List[numpy.ndarray]
        A list of coordinate sets to write as the airfoil geometry. The array of coordinates has size
        :math:`N \times 2` where :math:`N` is the number of airfoil
        coordinates.

    Returns
    -------
    bool, str, typing.List[str]
        A boolean describing whether the MSET call succeeded, a string containing the path to the MSET log file,
        and a list containing the order of the airfoil names determined by vertical position, descending order
    """
    if coords is None and mea is None:
        raise ValueError("Must specify either coords or mea")
    if coords is not None and mea is not None:
        raise ValueError("Cannot specify both coords and mea")

    if coords is not None:
        blade_file_path, airfoil_name_order = write_blade_file(name, base_dir, mset_settings['grid_bounds'], coords,
                                                               mea_airfoil_names=mea_airfoil_names)
    elif mea is not None:
        blade_file_path, airfoil_name_order = mea.write_mses_blade_file(
            name, os.path.join(base_dir, name), grid_bounds=mset_settings["grid_bounds"],
            max_airfoil_points=mset_settings["downsampling_max_pts"] if bool(mset_settings["use_downsampling"]) else None,
            curvature_exp=mset_settings["downsampling_curve_exp"])
    else:
        raise ValueError("At least one of either coords or mea must be specified to write the blade file")

    write_gridpar_file(name, base_dir, mset_settings, airfoil_name_order=airfoil_name_order)
    mset_input_name = 'mset_input.txt'
    mset_input_file = os.path.join(base_dir, name, mset_input_name)
    mset_input_list = ['1', '0', '2', '', '', '', '3', '4', '0']
    write_input_file(mset_input_file, mset_input_list)
    mset_log = os.path.join(base_dir, name, 'mset.log')
    with open(mset_log, 'wb') as f:
        with open(mset_input_file, 'r') as g:
            process = subprocess.Popen(['mset', name], stdin=g, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       cwd=os.path.join(base_dir, name), shell=False)
            try:
                outs, errs = process.communicate(timeout=mset_settings['timeout'])
                f.write('Output:\n'.encode('utf-8'))
                f.write(outs)
                f.write('\nErrors:\n'.encode('utf-8'))
                f.write(errs)
                mset_success = True
            except subprocess.TimeoutExpired:
                process.kill()
                outs, errs = process.communicate()
                f.write('After timeout, \nOutput: \n'.encode('utf-8'))
                f.write(outs)
                f.write('\nErrors:\n'.encode('utf-8'))
                f.write(errs)
                mset_success = False
    return mset_success, mset_log, airfoil_name_order


def run_mses(name: str, base_folder: str, mses_settings: dict, airfoil_name_order: typing.List[str],
             stencil: bool = False, conn: multiprocessing.connection.Connection = None) -> (bool, str):
    r"""
    A Python wrapper for MSES

    Parameters
    ----------
    name: str
        Name of the airfoil [system]

    base_folder: str
        MSES files will be stored in ``base_folder/name``

    mses_settings: dict
        Flow parameter set (dictionary)

    airfoil_name_order: typing.List[str]
        List of the names of the airfoils (from top to bottom)

    stencil: bool
        Whether a multipoint stencil is to be used. This variable is only used here to determine whether to overwrite or
        append to the log file. Default: ``False``

    conn: multiprocessing.connection.Connection
        Pipe used to transfer intermediate and final results to the GUI when this function is run from the GUI.
        This keyword argument should not be set if using this function directly.

    Returns
    -------
    bool, str
        A boolean describing whether the MSES solution is converged and a string containing the path to the MSES
        log file
    """

    def send_over_pipe(data: object):
        """
        Connection to the GUI that is only used if ``calculate_aero_data`` is being called directly from the GUI

        Parameters
        ----------
        data: object
            The intermediate information to pass to the GUI, normally a two-element tuple where the first argument
            is a string specifying the kind of data being sent, and the second argument being the actual data
            itself (note that the data must be picklable by the multiprocessing module)

        Returns
        -------

        """
        try:
            if conn is not None:
                conn.send(data)
        except BrokenPipeError:
            pass

    write_mses_file(name, base_folder, mses_settings, airfoil_name_order=airfoil_name_order)
    mses_log = os.path.join(base_folder, name, 'mses.log')
    if stencil:
        read_write = 'ab'
    else:
        read_write = 'wb'
    send_over_pipe(("clear_residual_plots", None))
    converged = False
    with open(mses_log, read_write) as f:
        process = subprocess.Popen(['mses', name, str(mses_settings['iter'])], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd=os.path.join(base_folder, name))
        try:
            if conn is not None:
                iteration, rms_dR, rms_dA, rms_dV = None, None, None, None
                for line in process.stdout:
                    decoded_line = line.decode("utf-8")
                    if "Converged" in decoded_line:
                        converged = True
                    if "Convergence failed." in decoded_line:
                        raise ConvergenceFailedError
                    f.write(line)
                    if "rms(dR):" in decoded_line:
                        decoded_line_split = decoded_line.split()
                        iteration = decoded_line_split[0]
                        rms_dR = decoded_line_split[decoded_line_split.index("rms(dR):") + 1]
                        if rms_dR == "NaN":
                            raise ConvergenceFailedError
                    elif "rms(dA):" in decoded_line:
                        decoded_line_split = decoded_line.split()
                        rms_dA = decoded_line_split[decoded_line_split.index("rms(dA):") + 1]
                    elif "rms(dV):" in decoded_line:
                        decoded_line_split = decoded_line.split()
                        rms_dV = decoded_line_split[decoded_line_split.index("rms(dV):") + 1]
                        send_over_pipe(("message", f"Iteration {iteration}: rms(dR) = {rms_dR}, rms(dA) = {rms_dA}, rms(dV) = {rms_dV}"))
                        send_over_pipe(("mses_residual", (int(iteration), float(rms_dR), float(rms_dA), float(rms_dV))))
            outs, errs = process.communicate(timeout=mses_settings['timeout'])

            if conn is None:
                if 'Converged' in str(outs):
                    converged = True
                    if mses_settings['verbose']:
                        print('Converged!')
                else:
                    if mses_settings['verbose']:
                        print('Not converged!')
                f.write('Output:\n'.encode('utf-8'))
                f.write(outs)
                f.write('\nErrors:\n'.encode('utf-8'))
                f.write(errs)
        except (subprocess.TimeoutExpired, ConvergenceFailedError):
            process.kill()
            outs, errs = process.communicate()
            f.write('After timeout, \nOutput: \n'.encode('utf-8'))
            f.write(outs)
            f.write('\nErrors:\n'.encode('utf-8'))
            f.write(errs)

    return converged, mses_log


def run_mplot(name: str, base_dir: str, mplot_settings: dict, mode: str = "forces", min_contour: float = 0.0,
              max_contour: float = 1.5, n_intervals: int = 0) -> str:
    r"""
    A Python wrapper for MPLOT

    Parameters
    ----------
    name: str
        Name of the airfoil [system]

    base_dir: str
        MSES files will be stored in ``base_folder/name``

    mplot_settings: dict
        Flow parameter set (dictionary)

    mode: str
        What type of data to output from MPLOT. Current choices are ``"forces"``, ``"Cp"``, ``"flowfield"``,
        ``"grid_zoom"``, ``"grid"``, ``"grid_stats"``, and ``"Mach contours"``. Default: ``"forces"``

    min_contour: float
        inimum contour level (only affects the result if ``mode=="Mach contours"``). Default: ``0.0``

    max_contour: float
        Maximum contour level (only affects the result if ``mode=="Mach contours"``). Default: ``1.5``

    n_intervals: int
        Number of contour levels (only affects the result if ``mode=="Mach contours"``). A value of ``0`` results in
        MPLOT automatically setting a "nice" value for the number of contour levels. Default: ``0``

    Returns
    -------
    str
        A string containing the path to the MPLOT log file
    """
    if mode in ["forces", "Forces"]:
        mplot_input_name = "mplot_forces_dump.txt"
        mplot_input_list = ['1', '12', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot.log')
    elif mode in ["CP", "cp", "Cp", "cP"]:
        mplot_input_name = "mplot_input_dumpcp.txt"
        mplot_input_list = ['12', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_cp.log')
    elif mode in ['flowfield', 'Flowfield', 'FlowField', 'flow_field']:
        mplot_input_name = "mplot_dump_flowfield.txt"
        mplot_input_list = ['11', '', 'Y', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_flowfield.log')
    elif mode in ['grid_zoom', 'Grid_Zoom', 'GRID_ZOOM']:
        mplot_input_name = "mplot_input_grid_zoom.txt"
        mplot_input_list = ['3', '1', '9', 'B', '', '6', '-0.441', '0.722', '1.937', '-0.626', '1',
                            '8', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_grid_zoom.log')
    elif mode in ['grid', 'Grid', 'GRID']:
        mplot_input_name = "mplot_input_grid.txt"
        mplot_input_list = ['3', '1', '8', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_grid.log')
    elif mode in ['grid_stats', 'GridStats', 'Grid_Stats']:
        mplot_input_name = "mplot_input_grid_stats.txt"
        mplot_input_list = ['3', '10', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_grid_stats.log')
    elif mode in ["Streamline_Grid"]:
        mplot_input_name = "mplot_streamline_grid.txt"
        mplot_input_list = ['10', '', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_streamline_grid.log')
    elif mode in ["Mach", "mach", "M", "m", "Mach contours", "Mach Contours", "mach contours"]:
        mplot_input_name = "mplot_inputMachContours.txt"
        if n_intervals == 0:
            mplot_input_list = ['3', '3', 'M', '', '9', 'B', '', '6', '-0.441', '0.722', '1.937', '-0.626', '3', 'M',
                                '', '8', '', '0']
        else:
            mplot_input_list = ['3', '3', 'M', '', '9', 'B', '', '6', '-0.441', '0.722', '1.937', '-0.626', '3', 'M',
                                f'{min_contour} {max_contour} {n_intervals}', '8', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_mach.log')
    else:
        raise Exception("Invalid MPLOT mode!")
    mplot_input_file = os.path.join(base_dir, name, mplot_input_name)
    write_input_file(mplot_input_file, mplot_input_list)

    mplot_attempts = 0
    mplot_max_attempts = 100
    while mplot_attempts < mplot_max_attempts:
        try:
            with open(mplot_log, 'wb') as f:
                with open(mplot_input_file, 'r') as g:
                    process = subprocess.Popen(['mplot', name], stdin=g, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                               cwd=os.path.join(base_dir, name))
                try:
                    outs, errs = process.communicate(timeout=mplot_settings['timeout'])
                    f.write('Output:\n'.encode('utf-8'))
                    f.write(outs)
                    f.write('\nErrors:\n'.encode('utf-8'))
                    f.write(errs)
                except subprocess.TimeoutExpired:
                    process.kill()
                    outs, errs = process.communicate()
                    f.write('After timeout, \nOutput: \n'.encode('utf-8'))
                    f.write(outs)
                    f.write('\nErrors:\n'.encode('utf-8'))
                    f.write(errs)
            break
        except OSError:
            # In case any of the log files cannot be created/read temporarily, wait a short period of time and try again
            time.sleep(0.01)
            mplot_attempts += 1

    if mode in ["Mach", "mach", "M", "m", "Mach contours", "Mach Contours", "mach contours"]:
        convert_ps_to_svg(os.path.join(base_dir, name),
                          'plot.ps',
                          'Mach_contours.pdf',
                          'Mach_contours.svg')
    elif mode in ['grid', 'Grid', 'GRID']:
        convert_ps_to_svg(os.path.join(base_dir, name), 'plot.ps', 'grid.pdf', 'grid.svg')
    elif mode in ['grid_zoom', 'Grid_Zoom', 'GRID_ZOOM']:
        convert_ps_to_svg(os.path.join(base_dir, name), 'plot.ps', 'grid_zoom.pdf', 'grid_zoom.svg')
    return mplot_log


def write_blade_file(name: str, base_dir: str, grid_bounds: typing.List[float], coords: typing.List[np.ndarray],
                     mea_airfoil_names: typing.List[str]) -> (str, typing.List[str]):
    r"""
    Writes airfoil geometry to an MSES blade file

    Parameters
    ----------
    name: str
        Name of the airfoil [system]

    base_dir: str
        Blade file will be stored as ``base_dir/name/blade.name``

    grid_bounds: typing.List[float]
        The far-field boundary locations for MSES, of the form
        ``[x_lower, x_upper, y_lower, y_upper]``. For example, ``[-6, 6, -5, 5]`` will produce a pseudo-rectangular
        far-field boundary with :math:`x` going from -6 to 6 and :math:`y` going from -5 to 5. The boundary is not
        perfectly rectangular because MSES produces far-field boundaries that follow the streamlines close to the
        specified :math:`x` and :math:`y`-locations for the grid.

    coords: typing.List[np.ndarray]
        A 3-D set of coordinates to write as the airfoil geometry. The array of coordinates has size
        :math:`M \times N \times 2` where :math:`M` is the number of airfoils and :math:`N` is the number of airfoil
        coordinates. The coordinates can be input as a ragged array, where :math:`N` changes with each 3-D slice (i.e.,
        the number of airfoil coordinates can be different for each airfoil).

    mea_airfoil_names: typing.List[str]
        List of airfoil names contained in the MEA

    Returns
    -------
    str, typing.List[str]
        Absolute path to the generated MSES blade file and a list of the airfoil names found in the MEA, ordered
        by vertical position (descending order)
    """

    # Set the default grid bounds value
    if grid_bounds is None:
        grid_bounds = [-5.0, 5.0, -5.0, 5.0]

    # Write the header (line 1: airfoil name, line 2: grid bounds values separated by spaces)
    header = name + "\n" + " ".join([str(gb) for gb in grid_bounds])

    # Determine the correct ordering for the airfoils. MSES expects airfoils to be ordered from top to bottom
    max_y = [np.max(coord_xy[:, 1]) for coord_xy in coords]
    airfoil_order = np.argsort(max_y)[::-1]

    # Loop through the airfoils in the correct order
    mea_coords = None
    for airfoil_idx in airfoil_order:
        airfoil_coords = coords[airfoil_idx]  # Extract the airfoil coordinates for this airfoil
        if mea_coords is None:
            mea_coords = airfoil_coords
        else:
            mea_coords = np.row_stack((mea_coords, np.array([999.0, 999.0])))  # MSES-specific airfoil delimiter
            mea_coords = np.row_stack((mea_coords, airfoil_coords))  # Append this airfoil's coordinates to the mat.

    # Generate the full file path
    blade_file_path = os.path.join(base_dir, name, f"blade.{name}")

    # Save the coordinates to file
    attempt = 0
    max_attempts = 100
    while attempt < max_attempts:
        try:
            np.savetxt(blade_file_path, mea_coords, header=header, comments="")
            break
        except OSError:
            time.sleep(0.01)
            attempt += 1

    # Get the airfoil name order
    airfoil_name_order = [mea_airfoil_names[idx] for idx in airfoil_order]

    return blade_file_path, airfoil_name_order


def write_gridpar_file(name: str, base_folder: str, mset_settings: dict, airfoil_name_order: typing.List[str]) -> str:
    """
    Writes grid parameters to a file readable by MSES

    Parameters
    ----------
    name: str
        Name of the airfoil [system]

    base_folder: str
        Grid parameter file will be stored as ``base_folder/name/gridpar.name``

    mset_settings: dict
        Parameter set (dictionary)

    airfoil_name_order: typing.List[str]
        List of airfoil names found in the MEA, ordered by vertical position (descending order). This order is output
        by ``write_blade_file``

    Returns
    -------
    str
        Path of the created grid parameter file
    """
    if not os.path.exists(os.path.join(base_folder, name)):  # if specified directory doesn't exist,
        os.mkdir(os.path.join(base_folder, name))  # create it
    gridpar_file = os.path.join(base_folder, name, 'gridpar.' + name)
    attempt = 0
    max_attempts = 100
    while attempt < max_attempts:
        try:
            with open(gridpar_file, 'w') as f:  # Open the gridpar_file with write permission
                f.write(f"{int(mset_settings['airfoil_side_points'])}\n")
                f.write(f"{mset_settings['exp_side_points']}\n")
                f.write(f"{int(mset_settings['inlet_pts_left_stream'])}\n")
                f.write(f"{int(mset_settings['outlet_pts_right_stream'])}\n")
                f.write(f"{int(mset_settings['num_streams_top'])}\n")
                f.write(f"{int(mset_settings['num_streams_bot'])}\n")
                f.write(f"{int(mset_settings['max_streams_between'])}\n")
                f.write(f"{mset_settings['elliptic_param']}\n")
                f.write(f"{mset_settings['stag_pt_aspect_ratio']}\n")
                f.write(f"{mset_settings['x_spacing_param']}\n")
                f.write(f"{mset_settings['alf0_stream_gen']}\n")

                multi_airfoil_grid = mset_settings['multi_airfoil_grid']

                for a in airfoil_name_order:
                    f.write(f"{multi_airfoil_grid[a]['dsLE_dsAvg']} {multi_airfoil_grid[a]['dsTE_dsAvg']} "
                            f"{multi_airfoil_grid[a]['curvature_exp']}\n")

                for a in airfoil_name_order:
                    f.write(f"{multi_airfoil_grid[a]['U_s_smax_min']} {multi_airfoil_grid[a]['U_s_smax_max']} "
                            f"{multi_airfoil_grid[a]['L_s_smax_min']} {multi_airfoil_grid[a]['L_s_smax_max']} "
                            f"{multi_airfoil_grid[a]['U_local_avg_spac_ratio']} {multi_airfoil_grid[a]['L_local_avg_spac_ratio']}\n")
            break
        except OSError:
            time.sleep(0.01)
            attempt += 1

    return gridpar_file


def write_mses_file(name: str, base_folder: str, mses_settings: dict, airfoil_name_order: typing.List[str]):
    """
    Writes MSES flow parameters to a file

    Parameters
    ==========
    name: str
      Name of the airfoil [system]

    base_folder: str
      MSES flow parameter file will be stored as ``base_folder/name/mses.name``

    mses_settings: dict
      Parameter set (dictionary)

    Returns
    =======
    str
      Path of the created MSES flow parameter file
    """
    F = deepcopy(mses_settings)
    if not os.path.exists(os.path.join(base_folder, name)):  # if specified directory doesn't exist,
        os.mkdir(os.path.join(base_folder, name))  # create it
    mses_file = os.path.join(base_folder, name, 'mses.' + name)

    # ============= Reynolds number calculation =====================
    if not bool(F['viscous_flag']):
        F['REYNIN'] = 0.0

    if F['inverse_side'] % 2 != 0:
        F['ISMOVE'] = 1
        F['ISPRES'] = 1
    elif F['inverse_side'] == 0:
        F['ISMOVE'] = 0
        F['ISPRES'] = 0
    else:
        F['ISMOVE'] = 2
        F['ISPRES'] = 2

    attempt = 0
    max_attempts = 100
    while attempt < max_attempts:
        try:
            with open(mses_file, 'w') as f:
                if F['target'] == 'alfa':
                    global_constraint_target = 5  # Tell MSES to specify the angle of attack in degrees given by 'ALFIN'
                elif F['target'] == 'Cl':
                    global_constraint_target = 6  # Tell MSES to target the lift coefficient specified by 'CLIFIN'
                else:
                    raise ValueError('Invalid value for \'target\' (must be either \'alfa\' or \'Cl\')')

                if not F['inverse_flag']:
                    f.write(f'3 4 5 7\n3 4 {global_constraint_target} 7\n')
                else:
                    f.write(f'3 4 5 7 11 12\n3 4 {global_constraint_target} 7 11 12\n')

                f.write(f"{F['MACHIN']} {F['CLIFIN']} {F['ALFAIN']}\n")
                f.write(f"{int(F['ISMOM'])} {int(F['IFFBC'])}\n")
                f.write(f"{F['REYNIN']} {F['ACRIT']}\n")

                for idx, airfoil_name in enumerate(airfoil_name_order):
                    f.write(f"{F['XTRSupper'][airfoil_name]} {F['XTRSlower'][airfoil_name]}")
                    if idx == len(airfoil_name_order) - 1:
                        f.write("\n")
                    else:
                        f.write(" ")

                f.write(f"{F['MCRIT']} {F['MUCON']}\n")

                if any([bool(flag) for flag in F['AD_flags']]) or bool(F['inverse_flag']):
                    f.write(f"{int(F['ISMOVE'])} {int(F['ISPRES'])}\n")
                    f.write(f"{int(F['NMODN'])} {int(F['NPOSN'])}\n")

                for idx, flag in enumerate(F['AD_flags']):
                    if bool(flag):
                        f.write(f"{int(F['ISDELH'][idx])} {F['XCDELH'][idx]} {F['PTRHIN'][idx]} {F['ETAH'][idx]}\n")
            break
        except OSError:
            time.sleep(0.01)
            attempt += 1

    mses_settings = F

    return mses_file, mses_settings


def write_input_file(input_file: str, input_list: typing.List[str]):
    """
    Writes inputs from a list to a file for use as STDIN commands to the shell/terminal.

    Parameters
    ==========
    input_file: str
      File where inputs are written

    input_list: typing.List[str]
      List of inputs to write. For example, passing ``["1", "", "12", "13"]`` is equivalent to typing the command
      sequence ``1, RETURN, RETURN, 12, RETURN, 13, RETURN`` into the shell or terminal.
    """
    attempt = 0
    max_attempts = 100
    while attempt < max_attempts:
        try:
            with open(input_file, 'w') as f:
                for input_ in input_list:
                    f.write(input_)
                    f.write('\n')
            break
        except OSError:
            time.sleep(0.01)
            attempt += 1


def convert_xfoil_string_to_aero_data(line1: str, line2: str, aero_data: dict):
    """
    Extracts aerodynamic data from strings pulled from XFOIL log files. The two string inputs are the string outputs
    from ``pymead.analysis.read_aero_data.read_aero_data_from_xfoil``

    Parameters
    ==========
    line1: str
      First line containing the aerodynamic data in the XFOIL log file.

    line2: str
      Second line containing the aerodynamic data in the XFOIL log file.

    aero_data: dict
      Dictionary to which to write the aerodynamic data
    """
    new_str = line1.replace(' ', '') + line2.replace(' ', '')
    new_str = new_str.replace('=>', '')
    appending = False
    data_list = []
    for ch in new_str:
        if ch.isdigit() or ch == '.' or ch == '-':
            if appending:
                data_list[-1] += ch
        else:
            appending = False
        last_ch = ch
        if last_ch == '=' and not ch == '>':
            appending = True
            data_list.append('')
    aero_data['alf'] = float(data_list[4])
    aero_data['Cm'] = float(data_list[0])
    aero_data['Cd'] = float(data_list[1])
    aero_data['Cdf'] = float(data_list[2])
    aero_data['Cdp'] = float(data_list[3])
    aero_data['Cl'] = float(data_list[5])
    aero_data['L/D'] = aero_data['Cl'] / aero_data['Cd']
    return aero_data


def line_integral_CPK_inviscid(Cp_up: np.ndarray, Cp_down: np.ndarray, rho_up: np.ndarray, rho_down: np.ndarray,
                               u_up: np.ndarray, u_down: np.ndarray,
                               v_up: np.ndarray, v_down: np.ndarray, V_up: np.ndarray, V_down: np.ndarray,
                               x: np.ndarray, y: np.ndarray) -> float:
    r"""
    Computes the mechanical flow power coefficient line integral in the inviscid streamtube only according to
    a normalized version of Eq. (33) in Drela's "Power Balance in Aerodynamic Flows", a 2009 AIAA Journal article

    Parameters
    ----------
    Cp_up: np.ndarray
        1-D array of pressure coefficient values at the cell-centers of the column of cells immediately
        upstream of the actuator disk

    Cp_down: np.ndarray
        1-D array of pressure coefficient values at the cell-centers of the column of cells immediately
        downstream of the actuator disk

    rho_up: np.ndarray
        1-D array of density values (:math:`\rho/\rho_\infty`) at the cell-centers of the column of cells immediately
        upstream of the actuator disk

    rho_down: np.ndarray
        1-D array of density values (:math:`\rho/\rho_\infty`) at the cell-centers of the column of cells immediately
        downstream of the actuator disk

    u_up: np.ndarray
        1-D array of :math:`x`-velocity values (:math:`u/V_\infty`) at the cell-centers of the column of cells
        immediately upstream of the actuator disk

    u_down: np.ndarray
        1-D array of :math:`x`-velocity values (:math:`u/V_\infty`) at the cell-centers of the column of cells
        immediately downstream of the actuator disk

    v_up: np.ndarray
        1-D array of :math:`y`-velocity values (:math:`v/V_\infty`) at the cell-centers of the column of cells
        immediately upstream of the actuator disk

    v_down: np.ndarray
        1-D array of :math:`y`-velocity values (:math:`v/V_\infty`) at the cell-centers of the column of cells
        immediately downstream of the actuator disk

    V_up: np.ndarray
        1-D array of velocity values (:math:`V/V_\infty`) at the cell-centers of the column of cells
        immediately upstream of the actuator disk

    V_down: np.ndarray
        1-D array of velocity values (:math:`V/V_\infty`) at the cell-centers of the column of cells
        immediately downstream of the actuator disk

    x: np.ndarray
        2-d array of :math:`x`-coordinates of size :math:`3 \times N`, where :math:`N` is the number of streamlines
        captured by this actuator disk. The middle row (``x[1, :]``) represents the grid points spanning the actuator
        disk. The first row (``x[0, :]``) represents the column of grid points immediately upstream of the actuator
        disk, and the last row (``x[2, :]``) represents the column of grid points immediately downstream of the actuator
        disk.

    y: np.ndarray
        2-d array of :math:`y`-coordinates of size :math:`3 \times N`, where :math:`N` is the number of streamlines
        captured by this actuator disk. The middle row (``x[1, :]``) represents the grid points spanning the actuator
        disk. The first row (``x[0, :]``) represents the column of grid points immediately upstream of the actuator
        disk, and the last row (``x[2, :]``) represents the column of grid points immediately downstream of the actuator
        disk.

    Returns
    -------
    float
        The contribution of this actuator disk to the mechanical flow power coefficient, :math:`C_{P_K}`
    """
    # Calculate edge center locations
    x_edge_center_up = np.array([(x1 + x2) / 2 for x1, x2 in zip(x[0, :], x[1, :])])
    x_edge_center_down = np.array([(x1 + x2) / 2 for x1, x2 in zip(x[1, :], x[2, :])])
    y_edge_center_up = np.array([(y1 + y2) / 2 for y1, y2 in zip(y[0, :], y[1, :])])
    y_edge_center_down = np.array([(y1 + y2) / 2 for y1, y2 in zip(y[1, :], y[2, :])])

    # Calculate normals
    dx_dy_up = []
    dx_dy_down = []
    for i in range(len(x_edge_center_up) - 1):
        dx_dy_up.append((x_edge_center_up[i + 1] - x_edge_center_up[i]) / (
                y_edge_center_up[i + 1] - y_edge_center_up[i]))
    for i in range(len(x_edge_center_down) - 1):
        dx_dy_down.append((x_edge_center_down[i + 1] - x_edge_center_down[i]) / (
                y_edge_center_down[i + 1] - y_edge_center_down[i]))
    dx_dy_up = np.array(dx_dy_up)
    dx_dy_down = np.array(dx_dy_down)

    angle_up = np.arctan2(1, dx_dy_up)
    angle_down = np.arctan2(1, dx_dy_down)

    perp_angle_up = -np.pi / 2
    perp_angle_down = np.pi / 2

    n_hat_up = np.column_stack((np.cos(angle_up + perp_angle_up), np.sin(angle_up + perp_angle_up)))
    n_hat_down = np.column_stack((np.cos(angle_down + perp_angle_down), np.sin(angle_down + perp_angle_down)))

    # Generate velocity vectors
    V_vec_up = np.column_stack((u_up, v_up))
    V_vec_down = np.column_stack((u_down, v_down))

    # Calculate dL_B/c_main
    dl_up = []
    dl_down = []

    for i in range(len(x_edge_center_up) - 1):
        dl_up.append(np.hypot(x_edge_center_up[i + 1] - x_edge_center_up[i], y_edge_center_up[i + 1] - y_edge_center_up[i]))
    for i in range(len(x_edge_center_down) - 1):
        dl_down.append(np.hypot(x_edge_center_down[i + 1] - x_edge_center_down[i], y_edge_center_down[i + 1] - y_edge_center_down[i]))

    dl_up = np.array(dl_up)
    dl_down = np.array(dl_down)

    dl_up = np.cumsum(dl_up)
    dl_down = np.cumsum(dl_down)

    # Compute the dot product V_vec * n_hat
    dot_product_up = np.array([np.dot(V_vec_i, n_hat_i) for V_vec_i, n_hat_i in zip(V_vec_up, n_hat_up)])
    dot_product_down = np.array([np.dot(V_vec_i, n_hat_i) for V_vec_i, n_hat_i in zip(V_vec_down, n_hat_down)])

    # Compute the integrand
    integrand_up = (rho_up * (1 - V_up**2) - Cp_up) * dot_product_up.flatten()
    integrand_down = (rho_down * (1 - V_down**2) - Cp_down) * dot_product_down.flatten()

    # Integrate
    integral_up = np.trapz(integrand_up, dl_up)
    integral_down = np.trapz(integrand_down, dl_down)
    integral = integral_up + integral_down

    return integral


def calculate_CPK_mses_inviscid_only(analysis_subdir: str) -> typing.Dict[str, float]:
    r"""
    Calculates the mechanical flower power coefficient input to the control volume across the airfoil system control
    surface. Assumes that the control surface wraps just around the actuator disk and that the normal vectors point
    into the propulsor. Also assumes that there is no change in the kinetic energy defect across the actuator disk
    (and thus no change in CPK due to the boundary layer). The contribution of all actuator disks specified in the
    analysis (``.mses`` file) are summed to produce the total :math:`C_{P_K}`.

    Parameters
    ----------
    analysis_subdir: str
        Directory where the MSES analysis was performed

    Returns
    -------
    typing.Dict[str, float]
        A dictionary with one key (``"CPK"``) and one value (the value of the mechanical flow power coefficient)
    """
    airfoil_system_name = os.path.split(analysis_subdir)[-1]
    field_file = os.path.join(analysis_subdir, f'field.{airfoil_system_name}')
    grid_stats_file = os.path.join(analysis_subdir, 'mplot_grid_stats.log')
    grid_file = os.path.join(analysis_subdir, f'grid.{airfoil_system_name}')
    mses_log_file = os.path.join(analysis_subdir, "mses.log")

    field = read_field_from_mses(field_file)
    grid_stats = read_grid_stats_from_mses(grid_stats_file)
    x_grid, y_grid = read_streamline_grid_from_mses(grid_file, grid_stats)
    data_AD = read_actuator_disk_data_mses(mses_log_file, grid_stats)

    CPK = 0.0

    for data in data_AD:
        Cp_up = field[flow_var_idx["Cp"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        Cp_down = field[flow_var_idx["Cp"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        rho_up = field[flow_var_idx["rho"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        rho_down = field[flow_var_idx["rho"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        u_up = field[flow_var_idx["u"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        u_down = field[flow_var_idx["u"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        v_up = field[flow_var_idx["v"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        v_down = field[flow_var_idx["v"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        V_up = field[flow_var_idx["V"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        V_down = field[flow_var_idx["V"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        x = x_grid[data["flow_section_idx"]][data["field_i_up"]:data["field_i_up"] + 3, :]
        y = y_grid[data["flow_section_idx"]][data["field_i_up"]:data["field_i_up"] + 3, :]
        CPK += line_integral_CPK_inviscid(Cp_up, Cp_down, rho_up, rho_down, u_up, u_down,
                                          v_up, v_down, V_up, V_down, x, y)

    if np.isnan(CPK):
        CPK = 1e9

    return {"CPK": CPK}


class GeometryError(Exception):
    pass


class ConvergenceFailedError(Exception):
    pass
