import os
import typing

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import numpy
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib import ticker as mticker
from cycler import cycler
from matplotlib.patches import Polygon, Patch
import numpy as np
from copy import deepcopy
import PIL
from pymoo.decomposition.asf import ASF

from pymead.optimization.objectives_and_constraints import Objective
from pymead.utils.read_write_files import load_data, save_data
from pymead.optimization.pop_chrom import Chromosome, Population
from pymead.post.mses_field import generate_field_matplotlib
from pymead.analysis.read_aero_data import read_bl_data_from_mses
from pymead.core.geometry_collection import GeometryCollection
from pymead.utils.transformations import rotate_matrix
from pymead.post.fonts_and_colors import ILLINI_ORANGE, ILLINI_BLUE, font
from pymead.post.plot_formatters import format_axis_scientific, show_save_fig, legend_entry_flip
from pymead import DATA_DIR


ylabel = {
            "Cd": r"Design $C_d$ (counts)",
            "Cl": r"Design $C_l$",
            "Cdh": r"Design $C_{d_h}$ (counts)",
            "Cdw": r"Design $C_{d_w}$ (counts)",
            "alf": r"Design $\alpha$ (deg)",
            "Cdv": r"Design $C_{d_v}$ (counts)",
            "Cdp": r"Design $C_{d_p}$ (counts)",
            "Cdf": r"Design $C_{d_f}$ (counts)",
            "Cm": r"Design $C_m$",
            "CPK": r"Design $C_{P_K}$",
            "capSS": r"Supersonic Capture Streamtube",
            "sf": r"Design $C_{F_{\parallel V_\infty}}$ (counts)",
            "sfmh": r"Design $C_{F_{\parallel V_\infty}} - C_{d_h}$ (counts)",
            "Edot": r"Design $C_{\dot{\mathcal{E}}}$",
            "diss_surf": r"Design $C_{\Phi_{\mathrm{surf}}}$",
            "diss_shock": r"Design $C_{\Phi_{\mathrm{shock}}}$",
        }

bl_matplotlib_labels = {
    "Cp": r"$C_p$",
    "delta*": r"$\delta^*$",
    "theta": r"$\theta$",
    "theta*": r"$\theta^*$"
}


def field_display_text(index: int, display_var: str or typing.List[str], display_val: float or typing.List[float]):
    title = {
        "Cd": r"C_{F_{\parallel V_\infty}}",
        "CPK": r"C_{P_K}"
    }

    def output_display_text(disp_var: str, disp_val: float):
        disp_text = {
            "Cd": fr"${title['Cd']} = {disp_val:.3f}$",
            "CPK": fr"${title['CPK']} = {disp_val:.3f}$"
        }
        if disp_var not in disp_text.keys():
            raise ValueError(f"Failed to find display_var {disp_var} in display text keys ({disp_text.keys()})")
        return disp_text[disp_var]

    if isinstance(display_var, str) and isinstance(display_val, float):
        return f"Gen. {index}: " + output_display_text(display_var, display_val)
    elif isinstance(display_var, list) and isinstance(display_val, list):
        texts_to_join = []
        for dvar, dval in zip(display_var, display_val):
            texts_to_join.append(output_display_text(dvar, dval))
        return f"Gen. {index}: " + ", ".join(texts_to_join)
    else:
        raise TypeError("display_var and display_val must be str and float or List[str] and List[float]")


class PostProcess:
    def __init__(self, analysis_dir: str, image_dir: str, post_process_force_file: str, jmea_file: str,
                 modify_param_dict_func: typing.Callable = None, underwing: bool = False,
                 modify_param_dict_kwargs: dict = None, weights: np.ndarray = None,
                 settings_json_file: str = None):
        if modify_param_dict_kwargs is None:
            modify_param_dict_kwargs = {}
        self.cbar = None
        self.coord_plot_handles = []
        self.text_handles = []
        self.solid_handles = []
        self.analysis_dir = analysis_dir
        self.image_dir = image_dir
        if not os.path.exists(self.image_dir):
            os.mkdir(self.image_dir)
        self.post_process_force_file = post_process_force_file
        self.jmea_file = jmea_file
        self.geo_col_dict = load_data(jmea_file)
        self.geo_col_dict['airfoil_graphs_active'] = False
        self.underwing = underwing
        self.param_dict = load_data(os.path.join(self.analysis_dir, 'param_dict.json'))
        self.pop_size = deepcopy(self.param_dict["population_size"])
        print(f"{self.pop_size = }")
        self.param_dict_original = deepcopy(self.param_dict)
        self.modify_param_dict_func = modify_param_dict_func
        if self.modify_param_dict_func is not None:
            self.modify_param_dict(**modify_param_dict_kwargs)
        self.chromosomes = []
        self.multi_objective = self.determine_multi_objective()
        self.weights = weights if weights is not None else np.array([[0.5, 0.5]])
        self.settings_json_file = settings_json_file
        self.settings = None
        if self.settings_json_file is not None:
            self.settings = load_data(self.settings_json_file)

    def determine_multi_objective(self):
        alg1 = load_data(os.path.join(self.analysis_dir, f"algorithm_gen_1.pkl"))
        if alg1.opt.get("F").shape[1] > 1:
            return True
        return False

    def modify_param_dict(self, *args, **kwargs):
        self.modify_param_dict_func(self.param_dict, *args, **kwargs)

    def get_max_gen(self):
        max_gen = 1
        while True:
            if os.path.exists(os.path.join(self.analysis_dir, f"algorithm_gen_{max_gen}.pkl")):
                max_gen += 1
            else:
                return max_gen - 1

    def get_X(self, alg_file: str, weight_idx: int = 0):
        alg = load_data(alg_file)
        if alg.opt is None:
            return None
        else:
            if self.multi_objective:
                decomp = ASF()
                F = alg.opt.get("F")
                # I = decomp.do(F, self.weights[weight_idx]).argmin()
                F2 = deepcopy(F)
                # Normalize the objective function values before calculating the decomposition
                max_f1 = np.max(F2[:, 0])
                max_f2 = np.max(F2[:, 1])
                F2[:, 0] *= 1.0 / max_f1
                F2[:, 1] *= 1.0 / max_f2
                I2 = decomp.do(F2, self.weights[weight_idx]).argmin()
                return alg.opt.get("X")[I2]
            else:
                return alg.opt.get("X").flatten()

    def get_F(self, alg_file: str or None, weight_idx: int = 0):
        alg = None
        if alg_file is not None:
            alg = load_data(alg_file)
        if alg is None:
            baseline_file = os.path.join(self.analysis_dir, "J_baseline.dat")
            if os.path.exists(baseline_file):
                F = np.loadtxt(baseline_file)
                print(f"1, {F = }")
            else:
                F = self.get_baseline_obj_fun_values()
                print(f"2, {F = }")
            return F
        else:
            if self.multi_objective:
                decomp = ASF()
                F = alg.opt.get("F")

                # Normalize the objective function values before calculating the decomposition
                max_f1 = np.max(F[:, 0])
                max_f2 = np.max(F[:, 1])
                F[:, 0] *= 1.0 / max_f1
                F[:, 1] *= 1.0 / max_f2
                I_ = decomp.do(F, self.weights[weight_idx]).argmin()
                return alg.opt.get("F")[I_]
            else:
                return alg.opt.get("F").flatten()

    def get_full_gen_opt_F(self, alg_file: str or None):
        alg = None
        if alg_file is not None:
            alg = load_data(alg_file)
        if alg is None:
            baseline_file = os.path.join(self.analysis_dir, "J_baseline.dat")
            if os.path.exists(baseline_file):
                F = np.loadtxt(baseline_file)
            else:
                F = self.get_baseline_obj_fun_values()
            return F
        else:
            return alg.opt.get("F")

    def get_full_gen_pop_F(self, alg_file: str or None):
        alg = None
        if alg_file is not None:
            alg = load_data(alg_file)
        if alg is None:
            baseline_file = os.path.join(self.analysis_dir, "J_baseline.dat")
            if os.path.exists(baseline_file):
                F = np.loadtxt(baseline_file)
            else:
                F = self.get_baseline_obj_fun_values()
            return F
        else:
            return alg.pop.get("F")

    def get_mea(self, index: int, weight_idx: int = 0):
        if index != 0:
            X = self.get_X(os.path.join(self.analysis_dir, f"algorithm_gen_{index}.pkl"), weight_idx=weight_idx)
        else:
            X = None

        mea_object = GeometryCollection.set_from_dict_rep(self.geo_col_dict)

        if X is not None:
            mea_object.update_parameters(X)

        return mea_object

    def set_index(self, index: int or typing.Iterable = None):
        if isinstance(index, int):
            index = [index]
        elif index is None:
            index = np.arange(0, self.get_max_gen() + 1)
        return index

    def get_weight_str(self, weight_idx: int):
        weights = self.weights[weight_idx, :]
        if self.multi_objective:
            weight_str = "_w" + "".join([f"{int(100 * w)}-" for w in weights])[:-1]
        else:
            weight_str = ""
        return weight_str

    def get_baseline_obj_fun_values(self):
        if not os.path.exists(os.path.join(self.analysis_dir, 'analysis')):
            os.mkdir(os.path.join(self.analysis_dir, 'analysis'))

        param_set = deepcopy(self.param_dict_original)
        param_set['num_processors'] = 1
        param_set['population_size'] = 1
        param_set['mplot_settings']['flow_field'] = 2
        param_set['mses_settings']['timeout'] += 10.0
        param_set['mset_settings']['airfoil_analysis_dir'] = os.path.join(
            self.analysis_dir, 'analysis', f'analysis_0')
        param_set['mset_settings']['airfoil_coord_file_name'] = f'analysis_0'
        param_set['base_folder'] = os.path.join(self.analysis_dir, 'analysis')
        param_set['name'] = [f"analysis_0"]

        airfoil_name, mea_name = None, None
        if param_set["tool"] == "XFOIL":
            airfoil_name = param_set["xfoil_settings"]["airfoil"]
        elif param_set["tool"] == "MSES":
            mea_name = param_set["mset_settings"]["mea"]

        chromosome = Chromosome(param_dict=param_set, population_idx=0, geo_col_dict=deepcopy(self.geo_col_dict),
                                genes=None, generation=0, mea_name=mea_name, airfoil_name=airfoil_name)

        population = Population(param_dict=param_set, generation=0, parents=[chromosome],
                                verbose=True, skip_parent_assignment=False)

        population.eval_pop_fitness()
        population_forces = population.population[0].forces
        obj_fun_strings = self.settings["Genetic Algorithm"]["J"].split(",")
        J_baseline = []
        for J in obj_fun_strings:
            objective = Objective("")
            objective.set_func_str(J)
            objective.update(population_forces)
            J_baseline.append(objective.value)
        J_baseline = np.array(J_baseline)
        np.savetxt(os.path.join(self.analysis_dir, "J_baseline.dat"), J_baseline)
        return J_baseline

    def run_analysis(self, index: int or typing.Iterable = None, evaluate: bool = True, save_coords: bool = False,
                     save_control_points: bool = False, save_airfoil_state: bool = False):
        for weight_idx, weights in enumerate(self.weights):
            weight_str = self.get_weight_str(weight_idx)

            self.chromosomes = []

            index = self.set_index(index)

            X_list = [self.get_X(os.path.join(self.analysis_dir, f"algorithm_gen_{i}.pkl"), weight_idx=weight_idx)
                      if i != 0 else None for i in index]

            if not os.path.exists(os.path.join(self.analysis_dir, 'analysis')):
                os.mkdir(os.path.join(self.analysis_dir, 'analysis'))

            for idx, i in enumerate(index):
                param_set = deepcopy(self.param_dict)
                param_set['mset_settings']['airfoil_analysis_dir'] = os.path.join(
                    self.analysis_dir, 'analysis', f'analysis_{i}' + weight_str)
                param_set['mset_settings']['airfoil_coord_file_name'] = f'analysis_{i}' + weight_str
                param_set['base_folder'] = os.path.join(self.analysis_dir, 'analysis')
                param_set['name'] = [f"analysis_{j}" + weight_str for j in index]

                airfoil_name, mea_name = None, None
                if param_set["tool"] == "XFOIL":
                    airfoil_name = param_set["xfoil_settings"]["airfoil"]
                elif param_set["tool"] == "MSES":
                    mea_name = param_set["mset_settings"]["mea"]

                # parent_chromosomes.append(Chromosome(param_set=param_set, population_idx=s, mea=mea, X=X))
                X = X_list[idx]
                self.chromosomes.append(Chromosome(param_dict=param_set, population_idx=i,
                                                   geo_col_dict=deepcopy(self.geo_col_dict), genes=X,
                                                   generation=0, mea_name=mea_name, airfoil_name=airfoil_name))

            population = Population(param_dict=self.param_dict, generation=0, parents=self.chromosomes,
                                    verbose=True, skip_parent_assignment=False)
            population.generate_chromosomes_parallel()
            if save_coords:
                if not os.path.exists(os.path.join(self.analysis_dir, f'coords{weight_str}')):
                    os.mkdir(os.path.join(self.analysis_dir, f'coords{weight_str}'))
                for idx, c in enumerate(population.population):
                    save_data(c.coords, os.path.join(self.analysis_dir, f'coords{weight_str}', f'coords_{index[idx]}.json'))
            if save_control_points:
                if not os.path.exists(os.path.join(self.analysis_dir, f'control_points{weight_str}')):
                    os.mkdir(os.path.join(self.analysis_dir, f'control_points{weight_str}'))
                for idx, c in enumerate(population.population):
                    save_data(c.control_points, os.path.join(self.analysis_dir,
                                                             f'control_points{weight_str}', f'control_points_{index[idx]}.json'))
            if save_airfoil_state:
                if not os.path.exists(os.path.join(self.analysis_dir, f'airfoil_state{weight_str}')):
                    os.mkdir(os.path.join(self.analysis_dir, f'airfoil_state{weight_str}'))
                for idx, c in enumerate(population.population):
                    save_data(c.airfoil_state, os.path.join(self.analysis_dir,
                                                            f'airfoil_state{weight_str}', f'airfoil_state_{index[idx]}.json'))

            if evaluate:
                population.eval_pop_fitness()
                population_forces = {k: [c.forces[k] for c in population.population]
                                     for k in population.population[0].forces.keys()}
                post_process_file_split = os.path.splitext(self.post_process_force_file)
                save_data(population_forces, post_process_file_split[0] + weight_str + post_process_file_split[1])

    def generate_objective_function_plot(self, index: int or typing.Iterable = None):
        obj_fun_label_dict = {
            0: r"$J_P$",
            1: r"$J_F$"
        }
        save_name_dict = {
            0: "JP",
            1: "JF",
        }

        F_list = []
        for weight_idx, weights in enumerate(self.weights):
            index = self.set_index(index)
            F = np.array([self.get_F(os.path.join(self.analysis_dir, f"algorithm_gen_{i}.pkl"), weight_idx=weight_idx)
                          if i != 0 else self.get_F(alg_file=None, weight_idx=weight_idx) for i in index])
            F_list.append(F)
            print(f"{F = }")

        for obj_fun_idx in range(F_list[0].shape[1]):
            fig, axs = plt.subplots()

            prop_cycler = (cycler(color=['#004488', '#DDAA33', '#BB5566']))
            axs.set_prop_cycle(prop_cycler)

            for i, F in enumerate(F_list):
                weight_str = self.get_weight_str(i)
                label = None if self.weights.shape[0] < 2 else weight_str[1:]
                label_dict = {
                    "w0-100": f"POP{self.pop_size}L",
                    "w50-50": f"POP{self.pop_size}M",
                    "w100-0": f"POP{self.pop_size}R"
                }
                label = label_dict[label]

                # Determine what ydata to plot
                ydata = F[:, obj_fun_idx]

                # Plot the ydata
                axs.plot(ydata, label=label)

            fig.set_tight_layout('tight')
            if obj_fun_idx == 0 and self.pop_size == 150:
                axs.set_ylim([0.165, 0.235])
            axs.set_xlabel("Generation", fontdict=font)
            axs.set_ylabel(obj_fun_label_dict[obj_fun_idx], fontdict=font)
            axs.grid("on", ls=":")
            if self.weights.shape[0] > 1:
                legend_font_dict = {k: v for k, v in font.items() if k != "color"}
                axs.legend(prop=legend_font_dict)
            format_axis_scientific(axs)
            show_save_fig(fig, save_base_dir=self.image_dir, file_name_stub=f'{save_name_dict[obj_fun_idx]}')

    def generate_aero_force_plot(self, var: str or list = None):
        if self.multi_objective:
            weight_str = self.get_weight_str(0)
            post_process_file_split = os.path.splitext(self.post_process_force_file)
            post_process_forces = load_data(post_process_file_split[0] + weight_str + post_process_file_split[1])
            if self.weights.shape[0] > 1:
                post_process_forces_list = [load_data(
                    post_process_file_split[0] + self.get_weight_str(i) + post_process_file_split[1])
                    for i in range(self.weights.shape[0])]
            else:
                post_process_forces_list = [post_process_forces]
        else:
            post_process_forces = load_data(self.post_process_force_file)
            post_process_forces_list = [post_process_forces]
        if isinstance(var, str):
            var = [var]
        if var is None:
            var = [k for k in post_process_forces.keys() if k not in ['converged', 'BL', 'errored_out', 'timed_out',
                                                                      'epma']]
            var.extend(["sf", "sfmh"])
        multiplier = {
            "Cd": 10000,
            "Cl": 1,
            "Cdh": 10000,
            "Cdw": 10000,
            "alf": 1,
            "Cdv": 10000,
            "Cdp": 10000,
            "Cdf": 10000,
            "Cm": 1,
            "CPK": 1,
            "capSS": 1,
            "sf": 10000,
            "sfmh": 10000,
            "diss_shock": 1,
            "diss_surf": 1,
            "Edot": 1,
        }
        for v in var:
            fig, axs = plt.subplots()

            prop_cycler = (cycler(color=['#004488', '#DDAA33', '#BB5566']))
            axs.set_prop_cycle(prop_cycler)

            for i, ppf in enumerate(post_process_forces_list):
                weight_str = self.get_weight_str(i)
                label = None if self.weights.shape[0] < 2 else weight_str[1:]
                label_dict = {
                    "w0-100": f"POP{self.pop_size}L",
                    "w50-50": f"POP{self.pop_size}M",
                    "w100-0": f"POP{self.pop_size}R"
                }
                label = label_dict[label]

                # Determine what ydata to plot
                if v in ppf.keys():
                    if isinstance(post_process_forces['Cd'][0], typing.Iterable):
                        ydata = [el[1] * multiplier[v] for el in ppf[v]]
                    else:
                        ydata = [el * multiplier[v] for el in ppf[v]]
                elif v == "sf":
                    if isinstance(post_process_forces['Cd'][0], typing.Iterable):
                        ydata = [el[1] * multiplier[v] for el in ppf["Cd"]]
                    else:
                        ydata = [el * multiplier[v] for el in ppf["Cd"]]
                elif v == "sfmh":
                    if isinstance(post_process_forces['Cd'][0], typing.Iterable):
                        ydata = [(el_Cd[1] - el_Cdh[1]) * multiplier[v] for el_Cd, el_Cdh in zip(ppf["Cd"], ppf["Cdh"])]
                    else:
                        ydata = [(el_Cd - el_Cdh) * multiplier[v] for el_Cd, el_Cdh in zip(ppf["Cd"], ppf["Cdh"])]
                else:
                    raise ValueError(f"Found invalid key {v}")

                # Plot the ydata
                axs.plot(ydata, label=label)

            fig.set_tight_layout('tight')
            axs.set_xlabel("Generation", fontdict=font)
            axs.set_ylabel(ylabel[v], fontdict=font)
            axs.grid("on", ls=":")
            if self.weights.shape[0] > 1:
                legend_font_dict = {k: v for k, v in font.items() if k != "color"}
                axs.legend(prop=legend_font_dict)
            format_axis_scientific(axs)
            show_save_fig(fig, save_base_dir=self.image_dir, file_name_stub=f'design_{v}')

    def pareto_front(self, opt_start: int = 5, opt_end: int = None, opt_num: int = 10):

        fig, axs = plt.subplots(figsize=(8, 4.8))

        def generate_plots(ax: plt.Axes, opt_end_):
            index = self.set_index(None)
            if opt_end_ is None:
                opt_end_ = index[-1]
            all_opt_F = [self.get_full_gen_opt_F(os.path.join(self.analysis_dir, f"algorithm_gen_{i}.pkl"))
                     if i != 0 else self.get_full_gen_opt_F(alg_file=None) for i in index]
            all_pop_F = [self.get_full_gen_pop_F(os.path.join(self.analysis_dir, f"algorithm_gen_{i}.pkl"))
                         if i != 0 else self.get_full_gen_pop_F(alg_file=None) for i in index]
            opt_range = np.linspace(opt_start, opt_end_, opt_num, endpoint=True).astype(int)
            cmap = plt.get_cmap("plasma", len(opt_range))
            cmap_dict = {v: i for i, v in enumerate(opt_range)}

            pop_labeled = False
            opt_labeled = False
            for i, F in enumerate(all_pop_F):
                if i != 0:
                    extra_kwargs = {}
                    if not pop_labeled:
                        extra_kwargs["label"] = "Individual"
                        pop_labeled = True
                    valid_rows = np.argwhere(F[:, 0] < 1.0).flatten()
                    ax.plot(F[valid_rows, 0], F[valid_rows, 1], ls="none", marker="s", mfc="lightgray", mec="lightgray", alpha=0.3,
                             markersize=2, **extra_kwargs)
            for i, F in enumerate(all_opt_F):
                if F.ndim > 1:
                    sorted_order = np.argsort(F[:, 0], axis=None)
                    F[:, 0] = F[sorted_order, 0]
                    F[:, 1] = F[sorted_order, 1]
                star_kwargs = dict(ls="--", marker="*")
                gray_circle_kwargs = dict(ls="none", marker="o", mfc="none", mec="gray", markersize=3)
                base_kwargs = dict(ls="none", marker="x", mfc="red", mec="red")
                if i == 0:
                    axs.plot(F[0], F[1], **base_kwargs, label="Baseline")
                else:
                    if i in opt_range:
                        ax.plot(F[:, 0], F[:, 1], **star_kwargs, zorder=100, label=f"Opt, Gen {i}",
                                 color=cmap(cmap_dict[i]),
                                 mfc=cmap(cmap_dict[i]),
                                 mec=cmap(cmap_dict[i])
                                 )
                    else:
                        extra_kwargs = {}
                        if not opt_labeled:
                            extra_kwargs["label"] = "Optimal"
                            opt_labeled = True
                        ax.plot(F[:, 0], F[:, 1], **gray_circle_kwargs, **extra_kwargs)
            # ax.set_xlim([0.058, 0.19])  # FOR POP50 input ga_opt_70
            ax.set_xlim([0.17, 0.26])  # FOR POP50 consumption ga_opt_93
            # ax.set_xlim([0.17, 0.26])  # FOR POP150 consumption ga_opt_92
            ax.set_xlabel(r"$J_P$", fontdict=font)
            ax.set_ylabel(r"$J_F$", fontdict=font)
            format_axis_scientific(ax)

        generate_plots(axs, opt_end)
        legend_font_dict = {k: v for k, v in font.items() if k != "color"}
        legend_font_dict["size"] = 12
        axs.legend(prop=legend_font_dict, loc="upper right", ncol=1)

        axs_inset = axs.inset_axes([0.17, 0.53, 0.35, 0.4])
        generate_plots(axs_inset, opt_end)
        x1, x2, y1, y2 = 0.17, 0.20, 0.012, 0.014
        axs_inset.set_xlim(x1, x2)
        axs_inset.set_ylim(y1, y2)
        # x_labels = [item if idx % 2 else "" for idx, item in enumerate(axs_inset.get_xticklabels())]
        # y_labels = [item if not idx % 2 else "" for idx, item in enumerate(axs_inset.get_yticklabels())]
        # axs_inset.set_xticklabels(x_labels)
        # axs_inset.set_yticklabels(y_labels)
        axs_inset.xaxis.set_major_locator(mticker.MaxNLocator(2))
        axs_inset.yaxis.set_major_locator(mticker.MaxNLocator(2, prune="lower"))

        show_save_fig(fig, save_base_dir=self.image_dir, file_name_stub="pareto_front")

    def compare_geometries(self, index: list, plot_actuator_disk: bool = True,
                           rotate_x_axis_wind_direction: bool = False, rotate_zero_alf: bool = False):
        for weight_idx, weights in enumerate(self.weights):
            weight_str = self.get_weight_str(weight_idx)
            if len(index) != 2:
                raise ValueError(f"Comparison of only 2 geometries is supported. Current index list: {index}")
            fig, axs = plt.subplots(figsize=(12, 5))
            colors = [ILLINI_BLUE, ILLINI_ORANGE]
            save_filetypes = ['.svg', '.pdf']
            post_process_forces = None
            if rotate_x_axis_wind_direction:
                post_process_force_file = os.path.splitext(
                    self.post_process_force_file)[0] + weight_str + os.path.splitext(self.post_process_force_file)[-1]
                post_process_forces = load_data(post_process_force_file)
            for analysis_idx, i in enumerate(index):
                coords = load_data(os.path.join(self.analysis_dir, f'coords{weight_str}', f'coords_{i}.json'))
                alf = None
                if rotate_zero_alf:
                    dx = (coords[0][0][0] + coords[0][-1][0]) / 2
                    dy = (coords[0][0][1] + coords[0][-1][1]) / 2
                    alf = -np.arctan2(dy, dx)
                for airfoil in coords:
                    airfoil = np.array(airfoil)
                    if rotate_x_axis_wind_direction:
                        airfoil = rotate_matrix(airfoil, -np.deg2rad(post_process_forces["alf"][i][1]))
                    if rotate_zero_alf:
                        airfoil = rotate_matrix(airfoil, alf)
                    axs.plot(airfoil[:, 0], airfoil[:, 1], color=colors[analysis_idx], ls='dashed')
                if plot_actuator_disk:
                    control_points = load_data(os.path.join(self.analysis_dir,
                                                            f'control_points{weight_str}', f'control_points_{i}.json'))
                    # Index order: [airfoil][curve][point][x or y]
                    if self.underwing:
                        ad1_x = [control_points[1][2][0][0], control_points[0][3][0][0]]
                        ad1_y = [control_points[1][2][0][1], control_points[0][3][0][1]]
                        ad2_x = [control_points[2][1][0][0], control_points[1][4][0][0]]
                        ad2_y = [control_points[2][1][0][1], control_points[1][4][0][1]]
                    else:
                        ad1_x = [control_points[1][2][0][0], control_points[2][3][0][0]]
                        ad1_y = [control_points[1][2][0][1], control_points[2][3][0][1]]
                        ad2_x = [control_points[0][1][0][0], control_points[1][4][0][0]]
                        ad2_y = [control_points[0][1][0][1], control_points[1][4][0][1]]

                    ad1 = np.column_stack((np.array(ad1_x), np.array(ad1_y)))
                    ad2 = np.column_stack((np.array(ad2_x), np.array(ad2_y)))
                    if rotate_x_axis_wind_direction:
                        ad1 = rotate_matrix(ad1, -np.deg2rad(post_process_forces["alf"][i][1]))
                        ad2 = rotate_matrix(ad2, -np.deg2rad(post_process_forces["alf"][i][1]))
                    if rotate_zero_alf:
                        ad1 = rotate_matrix(ad1, alf)
                        ad2 = rotate_matrix(ad2, alf)
                    axs.plot(ad1[:, 0], ad1[:, 1], color=colors[analysis_idx], ls='dotted')
                    axs.plot(ad2[:, 0], ad2[:, 1], color=colors[analysis_idx], ls='dotted')

            legend_proxies = [Line2D([], [], color=c, ls='dashed') for c in colors]
            legend_names = ['initial', 'optimal']
            ncols = 1
            if plot_actuator_disk:
                legend_proxies.extend([Line2D([], [], color=c, ls='dotted') for c in colors])
                ncols += 1
                legend_names.extend(['initial AD', 'optimal AD'])

            if self.underwing:
                legend_loc = "lower left"
            else:
                legend_loc = "upper left"
            axs.legend(legend_proxies, legend_names, prop={'size': 18, 'family': font['family']},
                       fancybox=False, shadow=False, loc=legend_loc, ncol=ncols, frameon=False)
            axs.set_aspect('equal')
            axs.set_xlabel(r'$x/c_{main}$', fontdict=dict(family="serif", size=18))
            axs.set_ylabel(r'$y/c_{main}$', fontdict=dict(family="serif", size=18))
            axs.grid('on', ls=':')
            format_axis_scientific(axs)
            modifiers = ""
            if rotate_x_axis_wind_direction:
                modifiers += "_rotateWind"
            if rotate_zero_alf:
                modifiers += "_rotateZeroAlf"
            for ext in save_filetypes:
                fig.savefig(os.path.join(self.image_dir, f"geometry_{index[0]}_{index[1]}{modifiers}{weight_str}{ext}"),

                            bbox_inches="tight")

    def compare_camber_thickness(self, index: list, ls: tuple = (":", "--")):
        for weight_idx, weights in enumerate(self.weights):
            weight_str = self.get_weight_str(weight_idx)
            if len(index) != 2:
                raise ValueError(f"Comparison of only 2 geometries is supported. Current index list: {index}")

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            figs = [fig1, fig2]
            axs = [ax1, ax2]
            f_name = ["camber", "thickness"]
            xc_array = np.linspace(0, 1, 250)

            legend_titles = {
                "A0": ("Main (Base)", "Main (Opt)"),
                "A1": ("Hub (Base)", "Hub (Opt)"),
                "A2": ("Nac (Base)", "Nac (Opt)"),
            }

            prop_cycler = (cycler(color=['#004488', '#DDAA33', '#BB5566']))

            for ax in axs:
                ax.set_prop_cycle(prop_cycler)

            for i, geometry_idx in enumerate(index):
                mea_object = self.get_mea(geometry_idx, weight_idx=weight_idx)
                for a_name, a in mea_object.airfoils.items():
                    label = legend_titles[a_name][i]
                    plot_kwargs = dict(label=label, ls=ls[i])
                    tc = a.compute_thickness_at_points(xc_array)
                    hc = a.compute_camber_at_points(xc_array)
                    axs[0].plot(xc_array, hc, **plot_kwargs)
                    axs[1].plot(xc_array, tc, **plot_kwargs)

            ncols = 3
            for ax in axs:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(legend_entry_flip(handles, ncols), legend_entry_flip(labels, ncols),
                          loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=ncols, fancybox=False, shadow=False, frameon=False,
                          prop=dict(family="serif", size=15))
                format_axis_scientific(ax)

            axis_label_font = dict(family="serif", size=18)
            axs[0].set_xlabel("Chord Fraction", fontdict=axis_label_font)
            axs[1].set_xlabel("Chord Fraction", fontdict=axis_label_font)
            axs[0].set_ylabel("Camber-to-Chord Ratio", fontdict=axis_label_font)
            axs[1].set_ylabel("Thickness-to-Chord Ratio", fontdict=axis_label_font)

            for i, fig in enumerate(figs):
                show_save_fig(fig, save_base_dir=self.image_dir, file_name_stub=f_name[i] + weight_str)

    def compare_airfoil_and_camber_to_known_airfoil(self, airfoil: str, index: int):
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        figs = [fig1, fig2]
        axs = [ax1, ax2]
        f_name = [f"main_opt_{airfoil}_compare_coords", f"main_opt_{airfoil}_compare_camber"]
        xc_array = np.linspace(0, 1, 250)
        airfoil_coords = np.loadtxt(os.path.join(DATA_DIR, airfoil, f"{airfoil}.dat"))
        airfoil_camber = np.loadtxt(os.path.join(DATA_DIR, airfoil, f"{airfoil}_camber.dat"))
        legend_titles = ("Main (Opt)", airfoil)

        prop_cycler = (cycler(color=['#004488', '#DDAA33', '#BB5566']))

        for ax in axs:
            ax.set_prop_cycle(prop_cycler)

        mea_object = self.get_mea(index)
        a = mea_object.airfoils["A0"]
        hc = a.compute_camber_at_points(xc_array)
        opt_coords = mea_object.airfoils["A0"].get_coords(body_fixed_csys=True)
        axs[0].plot(opt_coords[:, 0], opt_coords[:, 1], label=legend_titles[0], ls="-.")
        axs[0].plot(airfoil_coords[:, 0], airfoil_coords[:, 1], label=legend_titles[1], ls="-.")
        axs[0].set_aspect("equal")
        axs[1].plot(xc_array, hc, label=legend_titles[0], ls="-.")
        axs[1].plot(airfoil_camber[:, 0], airfoil_camber[:, 1], label=legend_titles[1], ls="-.")
        axis_label_font = dict(family="serif", size=18)
        axs[0].set_xlabel(r"$x/c$", fontdict=axis_label_font)
        axs[1].set_xlabel(r"$x/c$", fontdict=axis_label_font)
        axs[0].set_ylabel(r"$y/c$", fontdict=axis_label_font)
        axs[1].set_ylabel(r"$h/c$", fontdict=axis_label_font)

        ncols = 2
        bbox = ((0.5, 1.8), (0.5, 1.22))
        for i, ax in enumerate(axs):
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(legend_entry_flip(handles, ncols), legend_entry_flip(labels, ncols),
                      loc='upper center', bbox_to_anchor=bbox[i], ncol=ncols, fancybox=False, shadow=False, frameon=False,
                      prop=dict(family="serif", size=15))
            format_axis_scientific(ax)

        for ax in axs:
            format_axis_scientific(ax)

        for i, fig in enumerate(figs):
            show_save_fig(fig, save_base_dir=self.image_dir, file_name_stub=f_name[i])

    def generate_BL_plot(self, var: str or typing.Iterable = None, index: int or typing.Iterable = None,
                         mode: str = 'standalone', legend_strs: tuple = ('upper', 'lower')):

        save_filetypes = ['.svg', '.pdf']
        pai_props = {
            0: dict(title='Main', xlim=[-0.1, 1.1], xlabel=r'$x/c_{main}$', ylabel=bl_matplotlib_labels[var]),
            1: dict(title='Hub', xlim=[0.75, 1.4], xlabel=r'$x/c_{main}$', ylabel=''),
            2: dict(title='Nacelle', xlim=[0.65, 1.1], xlabel=r'$x/c_{main}$', ylabel='')
        }

        valid_modes = ['compare', 'standalone']
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        index = self.set_index(index)
        if isinstance(index, int):
            index = [index]
        if isinstance(var, str):
            var = [var]
        BL_data = {}

        fig, axs = None, None
        for v in var:
            for weight_idx, w in enumerate(self.weights):
                weight_str = self.get_weight_str(weight_idx)
                for analysis_idx, i in enumerate(index):
                    bl_file = os.path.join(self.analysis_dir, 'analysis',
                                           f'analysis_{i}' + weight_str, f'bl.analysis_{i}' + weight_str)
                    BL_data[i] = read_bl_data_from_mses(bl_file)
                    n_sides = len(BL_data[i])
                    if mode == 'compare' and analysis_idx == 0:
                        fig, axs = plt.subplots(nrows=1, ncols=int(n_sides / 2), figsize=(12, 3.3))
                    elif mode == 'standalone':
                        fig, axs = plt.subplots(nrows=1, ncols=int(n_sides / 2), figsize=(12, 3.3))
                    colors = [ILLINI_BLUE, ILLINI_ORANGE]
                    line_styles = ['dashed', 'dotted']
                    for side in range(n_sides):
                        color = None
                        if mode == 'compare':
                            color = colors[analysis_idx]
                        elif mode == 'standalone':
                            color = colors[0]
                        if self.underwing:
                            target_axs = int(side / 2)
                        else:
                            target_axs = 2 - int(side / 2)
                        axs[target_axs].plot(BL_data[i][side]['x'], BL_data[i][side][v], color=color,
                                             ls=line_styles[side % 2])
                    for k, v_ in pai_props.items():
                        axs[k].set_xlim(v_["xlim"])
                        axs[k].set_title(v_["title"], fontdict=font)
                        axs[k].set_xlabel(v_["xlabel"], fontdict=font)
                        axs[k].set_ylabel(v_["ylabel"], fontdict=font)
                        if v == "Cp":
                            axs[k].set_yticks([-1, 0, 1])
                        axs[k].grid('on', ls=':')
                    for k in range(3):
                        pos = axs[k].get_position()
                        axs[k].set_position([pos.x0, pos.y0 * 1.35, pos.width, pos.height * 0.83])
                    if v == 'Cp' and analysis_idx == 0:
                        for j in range(int(n_sides / 2)):
                            axs[j].invert_yaxis()

                    for ax in axs:
                        format_axis_scientific(ax)

                    # Set the legend
                    if mode == 'compare':
                        legend_proxies = [Line2D([], [], ls=ls, color=c)
                                          for ls, c in tuple(zip(line_styles * 2,
                                                                 [colors[0], colors[0], colors[1], colors[1]]))]
                        fig.legend(legend_proxies, legend_strs, fancybox=False, shadow=False, frameon=False, bbox_to_anchor=(0.5, 0.97),
                                   ncol=len(legend_proxies), loc='upper center', prop={'size': 14,
                                                                                       'family': font['family']})

                    if mode == 'standalone':
                        for ext in save_filetypes:
                            fig.savefig(os.path.join(self.image_dir, f'gen_{i}_{v}{weight_str}{ext}.svg'.replace('*', 'star')),
                                        bbox_inches="tight")
                if mode == 'compare':
                    for ext in save_filetypes:
                        fig.savefig(os.path.join(self.image_dir,
                                                 f'gen_{index[0]}_{index[1]}_{v}{weight_str}{ext}'.replace('*', 'star')),
                                    bbox_inches="tight")

    def generate_field_matplotlib(self, var: str, axs: plt.Axes or None, index: int, cmap_field: mpl_colors.Colormap or str,
                                  cmap_airfoil: mpl_colors.Colormap or str, shading: str = 'gouraud', vmin: float = None,
                                  vmax: float = None, weight_idx: int = 0):
        return generate_field_matplotlib(axs=axs,
                                         analysis_subdir=os.path.join(self.analysis_dir,
                                                                      'analysis', f'analysis_{index}{self.get_weight_str(weight_idx)}'),
                                         var=var, cmap_field=cmap_field, cmap_airfoil=cmap_airfoil, shading=shading,
                                         vmin=vmin, vmax=vmax)

    def generate_single_field(self, var: str, index: int, index_list, cmap_field: mpl_colors.Colormap or str,
                              cmap_airfoil: mpl_colors.Colormap or str, shading: str = 'gouraud', vmin: float = None,
                              vmax: float = None, image_extensions: tuple = ('.png', '.pdf'),
                              field_display_var: str or typing.List[str] = "Cd", weight_idx_list: list = None):
        for weight_idx, weights in enumerate(self.weights):

            if weight_idx_list is not None and weight_idx not in weight_idx_list:
                continue

            weight_str = self.get_weight_str(weight_idx)
            airfoil_color = 'black'
            flow_var_label_matplotlib = {'M': r'Mach Number',
                                         'Cp': r'Pressure Coefficient',
                                         'p': r'Static Pressure ($p/p_\infty$)',
                                         'rho': r'Density ($\rho/\rho_\infty$)',
                                         'u': r'Velocity-x ($u/V_\infty$)',
                                         'v': r'Velocity-y ($v/V_\infty$)',
                                         'q': r'Speed of Sound ($q/V_\infty$)',
                                         "Cpt": r"Total Pressure Over P_inf",
                                         "dCpt": r"Delta Total Pressure",
                                         "dCp": r"Delta Pressure Coefficient"}
            post_process_forces = load_data(
                os.path.splitext(
                    self.post_process_force_file)[0] + weight_str + os.path.splitext(self.post_process_force_file)[1])
            fig, axs = plt.subplots(figsize=(10, 5))
            # plt.subplots_adjust(left=0.08, bottom=-0.1, right=1.05, top=1.2, wspace=0.0, hspace=0.0)
            quad = generate_field_matplotlib(axs=axs,
                                      analysis_subdir=os.path.join(self.analysis_dir,
                                                                   'analysis', f'analysis_{index}{weight_str}'),
                                      var=var, cmap_field=cmap_field, cmap_airfoil=cmap_airfoil, shading=shading,
                                      vmin=vmin, vmax=vmax)

            # Plot the airfoils
            coords = load_data(os.path.join(self.analysis_dir, f'coords{weight_str}', f'coords_{index}.json'))
            for airfoil in coords:
                airfoil = np.array(airfoil)
                axs.plot(airfoil[:, 0], airfoil[:, 1], color=airfoil_color)
                polygon = Polygon(airfoil, closed=False, color="#000000AA")
                axs.add_patch(polygon)

            # Get location of index in gen_list
            if isinstance(index_list, list):
                gen_index = index_list.index(index)
            elif isinstance(index_list, numpy.ndarray):
                gen_index = index_list.tolist().index(index)
            else:
                raise TypeError("index_list must be a list or a numpy ndarray")

            # Generate performance characteristic text
            if isinstance(field_display_var, str):
                if not isinstance(post_process_forces[field_display_var][gen_index], typing.Iterable):
                    display_val = post_process_forces[field_display_var][gen_index]
                else:
                    display_val = post_process_forces[field_display_var][gen_index][1]
            elif isinstance(field_display_var, list):
                display_val = []
                for fdv in field_display_var:
                    if not isinstance(post_process_forces[fdv][gen_index], typing.Iterable):
                        display_val.append(post_process_forces[fdv][gen_index])
                    else:
                        display_val.append(post_process_forces[fdv][gen_index][1])
            else:
                raise TypeError("field_display_var must be either str or list")

            axs.text(x=-0.15, y=0.32, s=field_display_text(index, field_display_var, display_val),
                     fontdict=dict(size=18, family='serif'))

            # Make the color bar
            field_key = "field_0" if "field_0" in quad[1].keys() else "field"
            cbar = fig.colorbar(quad[1][field_key], ax=axs, shrink=0.7)
            cbar.ax.set_ylabel(flow_var_label_matplotlib[var], fontdict=dict(size=18, family='serif'))
            cbar.ax.tick_params(axis='y', which='major', labelsize=14)

            # Set various plot parameters
            axs.set_xlim([-0.2, 1.6])
            axs.set_ylim([-0.4, 0.4])
            format_axis_scientific(axs)
            axs.set_aspect('equal')

            # Label the axes
            axs.set_xlabel(r"$x/c_{main}$", fontdict=dict(size=18, family='serif'))
            axs.set_ylabel(r"$y/c_{main}$", fontdict=dict(size=18, family='serif'))

            # Generate the legend
            proxy_line = Line2D([], [], color=airfoil_color)
            proxy_patch_airfoil = Patch(color="#606060")
            proxy_patch_bl = Patch(color="white")
            proxy_arrow = Line2D([], [], color="blue")
            if self.underwing:
                legend_loc = "upper right"
            else:
                legend_loc = "lower right"
            axs.legend((proxy_line, proxy_patch_airfoil, proxy_patch_bl, proxy_arrow),
                       ('Surface', 'Solid', 'B.L.', 'Stag. S.L.'),
                       prop={'size': 17, 'family': font['family']}, loc=legend_loc, frameon=False)

            # Plot outputs
            fig.set_tight_layout('tight')
            fig_fname_no_ext = os.path.join(self.image_dir, f'field_{index}_{var}{weight_str}')
            for ext in image_extensions:
                kwargs = {'bbox_inches': 'tight'}
                if ext == '.png':
                    kwargs['dpi'] = 600
                fig.savefig(fig_fname_no_ext + ext, **kwargs)

    def generate_gif_from_images(self, var: str, index: int or typing.Iterable = None,
                                 save_name: str = 'opt_history.gif', duration: int = 150, weight_idx: int = 0):
        """Borrowed from https://pythonprogramming.altervista.org/png-to-gif/"""
        index = self.set_index(index)
        weight_str = self.get_weight_str(weight_idx)
        frames = []
        for i in index:
            image = os.path.join(self.image_dir, f'field_{i}_{var}{weight_str}.png')
            new_frame = PIL.Image.open(image)
            frames.append(new_frame)

        # Save into a GIF file that loops forever
        frames[0].save(os.path.join(self.image_dir, save_name), format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=duration, loop=0)

    def generate_field_gif_matplotlib(self, var: str,
                                      index: int or typing.Iterable = None, shading: str = 'gouraud'):

        flow_var_label_matplotlib = {'M': r'Mach Number',
                                     'Cp': r'Pressure Coefficient',
                                     'p': r'Static Pressure ($p/p_\infty$)',
                                     'rho': r'Density ($\rho/\rho_\infty$)',
                                     'u': r'Velocity-x ($u/V_\infty$)',
                                     'v': r'Velocity-y ($v/V_\infty$)',
                                     'q': r'Speed of Sound ($q/V_\infty$)',
                                     "Cpt": "Total Pressure Coefficient"}

        airfoil_color = 'black'

        post_process_forces = load_data(self.post_process_force_file)

        index = self.set_index(index)

        def animate(i):
            for h in self.coord_plot_handles:
                h_ = h.pop(0)
                h_.remove()
            for h in self.text_handles:
                h.remove()
            for h in self.solid_handles:
                h.remove()
            quad = self.generate_field_matplotlib(var=var, axs=None, index=i, cmap_field="Spectral_r",
                                                  cmap_airfoil=mpl_colors.ListedColormap("gray"), shading=shading,
                                                  vmin=0.0, vmax=1.4)
            coords = load_data(os.path.join(self.analysis_dir, 'coords', f'coords_{i}.json'))
            self.coord_plot_handles = []
            self.solid_handles = []
            self.text_handles = []
            for airfoil in coords:
                airfoil = np.array(airfoil)
                self.coord_plot_handles.append(axs.plot(airfoil[:, 0], airfoil[:, 1], color=airfoil_color))
                polygon = Polygon(airfoil, closed=False, color="#000000AA")
                self.solid_handles.append(axs.add_patch(polygon))
            self.text_handles.append(
                axs.text(x=-0.15, y=0.32, s=fr"Gen. {i}: $C_d = {post_process_forces['Cd'][i][1] * 10000:.1f}$ counts",
                         fontdict=font))
            # TODO: make sure that the color bar updates with each frame if vmin and vmax are set to None ('auto')
            if self.cbar is None:
                # divider = make_axes_locatable(axs)
                # cax = divider.append_axes("right", size="5%", pad=0.15)
                self.cbar = fig.colorbar(quad['field'], ax=axs, shrink=0.6)
                self.cbar.ax.set_ylabel(flow_var_label_matplotlib[var], fontdict=font)

        fig, axs = plt.subplots(figsize=(11, 3.3))
        plt.subplots_adjust(left=0.08, bottom=-0.1, right=1.05, top=1.2, wspace=0.0, hspace=0.0)
        axs.set_xlim([-0.2, 1.8])
        axs.set_ylim([-0.2, 0.4])
        axs.set_aspect('equal')
        axs.set_xlabel(r"$x/c_{main}$", fontdict=font)
        axs.set_ylabel(r"$y/c_{main}$", fontdict=font)
        proxy_line = Line2D([], [], color=airfoil_color)
        proxy_patch_airfoil = Patch(color="#000000DD")
        proxy_patch_bl = Patch(color="#606060")
        axs.legend((proxy_line, proxy_patch_airfoil, proxy_patch_bl), ('Surface', 'Solid', 'BL/Stag.'))

        anim = animation.FuncAnimation(fig, animate, frames=index, blit=False)

        # anim.save(os.path.join(self.image_dir, f'quad_animation_{var}_{shading}_shading.mp4'),
        #           writer=animation.FFMpegWriter(fps=10), dpi=450)

        anim.save(os.path.join(self.image_dir, f'quad_animation_{var}_{shading}_shading.gif'),
                  writer=animation.PillowWriter(fps=10))
