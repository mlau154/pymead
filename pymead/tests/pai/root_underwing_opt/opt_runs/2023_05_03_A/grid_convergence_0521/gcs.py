import os

import matplotlib.pyplot as plt
import pandas as pd

from pymead.analysis.read_aero_data import read_grid_stats_from_mses
from pymead.utils.read_write_files import load_data
from pymead.post.plot_formatters import format_axis_scientific, show_save_fig


def main():
    grid_names = ["ultracoarse3", "ultracoarse", "coarsest",
                  "coarse", "medium", "fine", "finest", "ultrafine", "ultrafine1", "ultrafine2"]
    data = {
        "airfoil_side_points": [],
        "num_streams_top": [],
        "max_streams_between": [],
        "num_streams_bot": [],
        "num_streamwise_cells": [],
        "num_streamnormal_cells": [],
        "num_cells_total": [],
        "alf": [],
        "Cd": [],
        "Cm": [],
        "CPK": [],
        "Edot": [],
        "diss_surf": [],
        "diss_shock": [],
        "grid_names": grid_names
    }
    for grid_name in grid_names:
        # Load grid stats, aero data, and settings files
        analysis_dir = "gc_" + grid_name
        grid_stats_log = os.path.join(analysis_dir, "mplot_grid_stats.log")
        aero_data_file = os.path.join(analysis_dir, "aero_data.json")
        settings_file = "gc_" + grid_name + "_settings.json"
        grid_stats = read_grid_stats_from_mses(grid_stats_log)
        aero_data = load_data(aero_data_file)
        settings = load_data(settings_file)

        # Write the grid settings data that was changed for each grid refinement level
        for var in ["airfoil_side_points", "num_streams_bot", "num_streams_top", "max_streams_between"]:
            data[var].append(settings["MSET"][var])

        # Write the output grid data
        data["num_streamwise_cells"].append(grid_stats["grid_size"][0])
        data["num_streamnormal_cells"].append(grid_stats["grid_size"][1])
        data["num_cells_total"].append(data["num_streamwise_cells"][-1] * data["num_streamnormal_cells"][-1])

        # Write the aero data
        for var in ["alf", "Cd", "Cm", "CPK", "Edot", "diss_surf", "diss_shock"]:
            data[var].append(aero_data[var])

    Cd04 = data["Cd"][4]
    Cd09 = data["Cd"][9]
    alf04 = data["alf"][4]
    alf09 = data["alf"][9]
    CPK04 = data["CPK"][4]
    CPK09 = data["CPK"][9]

    percdiffCd = 2 * (Cd04 - Cd09) / (Cd04 + Cd09) * 100
    percdiffalf = 2 * (alf04 - alf09) / (alf04 + alf09) * 100
    percdiffCPK = 2 * (CPK04 - CPK09) / (CPK04 + CPK09) * 100
    print(f"{percdiffCd = }, {percdiffalf = }, {percdiffCPK = }")

    df = pd.DataFrame(data=data)
    print(df.to_latex(index=False, columns=["grid_names", "airfoil_side_points", "num_streams_top",
                                            "max_streams_between", "num_streams_bot",
                                            "num_streamwise_cells", "num_streamnormal_cells", "num_cells_total"],
                      header=["Mesh", "Airfoil side points", r"$N_\text{SL}$ (top)", r"$N_\text{SL}$ (between, max)",
                              r"$N_\text{SL}$ (bottom)", "Streamwise cells", "Stream-normal cells", "Total cells"]))

    print(df.to_clipboard(columns=["grid_names", "airfoil_side_points", "num_streams_top",
                                            "max_streams_between", "num_streams_bot",
                                            "num_streamwise_cells", "num_streamnormal_cells", "num_cells_total"],
                      header=["Mesh", "Airfoil side points", r"$N_\text{SL}$ (top)", r"$N_\text{SL}$ (between, max)",
                              r"$N_\text{SL}$ (bottom)", "Streamwise cells", "Stream-normal cells", "Total cells"]))

    plt.rcParams["font.family"] = "serif"
    for var in ["alf", "Cd", "CPK"]:
        ylabels = {"alf": r"$\alpha$ (deg)", "Cd": r"$C_{F_{\parallel V_\infty}}$", "CPK": r"$C_{P_K}$"}
        fig, ax = plt.subplots()
        ax.grid(which='major', color='#99999955', linestyle=':', linewidth='0.2')
        # ax.minorticks_on()
        # ax.grid(which='minor', color='black', linestyle=':', linewidth='0.2')
        # ax.set_xlabel(r"$H=N^{-2/3}$", fontdict={"size": 20})
        ax.set_xlabel(r"$N_{cells}$", fontdict={"size": 20})
        ax.set_ylabel(ylabels[var], fontdict={"size": 20})
        # H = [N**(-2/3) for N in data["num_cells_total"]]
        H = [N for N in data["num_cells_total"]]
        ax.plot(H, data[var], color="steelblue", marker="s", mec="black", mfc="steelblue")
        ax.ticklabel_format(style="sci", axis="x", scilimits=(4, 5))
        format_axis_scientific(ax)
        fig.set_tight_layout("tight")
        show_save_fig(fig, ".", f"gcs_{var}", show=True, save=True)

    pass


if __name__ == "__main__":
    main()
