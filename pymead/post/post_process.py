import os
import typing

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
from matplotlib import animation
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon, Patch
import numpy as np
from copy import deepcopy
import PIL

from pymead.utils.read_write_files import load_data, save_data
from pymead.optimization.pop_chrom import Chromosome, Population
from pymead.post.mses_field import generate_field_matplotlib, flow_var_label
from pymead.analysis.read_aero_data import read_bl_data_from_mses
from functools import partial
from pymead.core.bezier import Bezier
from pymead.core.transformation import Transformation2D
from pymead.post.fonts_and_colors import ILLINI_ORANGE, ILLINI_BLUE, font


ylabel = {
            "Cd": r"Design $C_d$ (counts)",
            "Cl": r"Design $C_l$",
            "Cdh": r"Design $C_{d_h}$ (counts)",
            "Cdw": r"Design $C_{d_w}$ (counts)",
            "alf": r"Design $\alpha$ (deg)",
            "Cdv": r"Design $C_{d_v}$ (counts)",
            "Cdp": r"Design $C_{d_p}$ (counts)",
            "Cdf": r"Design $C_{d_f}$ (counts)",
            "Cm": r"Design $C_m$"
        }

bl_matplotlib_labels = {
    "Cp": r"$C_p$",
    "delta*": r"$\delta^*$",
    "theta": r"$\theta$",
}


class PostProcess:
    def __init__(self, analysis_dir: str, image_dir: str, post_process_force_file: str, jmea_file: str,
                 modify_param_dict_func: typing.Callable = None, underwing: bool = False):
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
        self.mea = load_data(jmea_file)
        self.mea['airfoil_graphs_active'] = False
        self.underwing = underwing
        self.param_dict = load_data(os.path.join(self.analysis_dir, 'param_dict.json'))
        self.modify_param_dict_func = modify_param_dict_func
        if self.modify_param_dict_func is not None:
            self.modify_param_dict()
        self.chromosomes = []

    def modify_param_dict(self):
        self.modify_param_dict_func(self.param_dict)

    def get_max_gen(self):
        max_gen = 0
        while True:
            if os.path.exists(os.path.join(self.analysis_dir, f"algorithm_gen_{max_gen}.pkl")):
                max_gen += 1
            else:
                return max_gen - 1

    @staticmethod
    def get_X(alg_file: str):
        alg = load_data(alg_file)
        if alg.opt is None:
            return None
        else:
            return alg.opt.get("X").flatten()

    def set_index(self, index: int or typing.Iterable = None):
        if isinstance(index, int):
            index = [index]
        elif index is None:
            index = np.arange(0, self.get_max_gen() + 1)
        return index

    def run_analysis(self, index: int or typing.Iterable = None, evaluate: bool = True, save_coords: bool = False,
                     save_control_points: bool = False, save_internal_radius: bool = False,
                     save_airfoil_state: bool = False):
        index = self.set_index(index)

        X_list = [self.get_X(os.path.join(self.analysis_dir, f"algorithm_gen_{i}.pkl")) for i in index]

        if not os.path.exists(os.path.join(self.analysis_dir, 'analysis')):
            os.mkdir(os.path.join(self.analysis_dir, 'analysis'))

        for i in index:
            param_set = deepcopy(self.param_dict)
            param_set['mset_settings']['airfoil_analysis_dir'] = os.path.join(
                self.analysis_dir, 'analysis', f'analysis_{i}')
            param_set['mset_settings']['airfoil_coord_file_name'] = f'analysis_{i}'
            param_set['base_folder'] = os.path.join(self.analysis_dir, 'analysis')
            param_set['name'] = [f"analysis_{j}" for j in index]

            # parent_chromosomes.append(Chromosome(param_set=param_set, population_idx=s, mea=mea, X=X))
            self.chromosomes.append(Chromosome(param_dict=param_set, population_idx=i, mea=self.mea, genes=X_list[i],
                                               ga_settings=None, category=None, generation=0))

        population = Population(param_dict=self.param_dict, ga_settings=None, generation=0, parents=self.chromosomes,
                                mea=self.mea, verbose=True, skip_parent_assignment=False)
        population.generate_chromosomes_parallel()
        if save_coords:
            if not os.path.exists(os.path.join(self.analysis_dir, 'coords')):
                os.mkdir(os.path.join(self.analysis_dir, 'coords'))
            for idx, c in enumerate(population.population):
                save_data(c.coords, os.path.join(self.analysis_dir, 'coords', f'coords_{idx}.json'))
        if save_control_points:
            if not os.path.exists(os.path.join(self.analysis_dir, 'control_points')):
                os.mkdir(os.path.join(self.analysis_dir, 'control_points'))
            for idx, c in enumerate(population.population):
                save_data(c.control_points, os.path.join(self.analysis_dir,
                                                         'control_points', f'control_points_{idx}.json'))
        if save_airfoil_state:
            if not os.path.exists(os.path.join(self.analysis_dir, 'airfoil_state')):
                os.mkdir(os.path.join(self.analysis_dir, 'airfoil_state'))
            for idx, c in enumerate(population.population):
                save_data(c.airfoil_state, os.path.join(self.analysis_dir,
                                                        'airfoil_state', f'airfoil_state_{idx}.json'))
        if save_internal_radius:
            if not os.path.exists(os.path.join(self.analysis_dir, 'radius')):
                os.mkdir(os.path.join(self.analysis_dir, 'radius'))
            for idx, c in enumerate(population.population):
                transformation = Transformation2D(tx=[-c.airfoil_state['A1']['dx']], ty=[-c.airfoil_state['A1']['dy']],
                                                  r=[-c.airfoil_state['A1']['alf']], order='t,r')
                nt = 300
                if self.underwing:
                    b_nac_fore = Bezier(P=np.array(c.control_points[0][2]), nt=nt)
                    b_nac_aft = Bezier(P=np.array(c.control_points[0][3]), nt=nt)
                else:
                    b_nac_fore = Bezier(P=np.array(c.control_points[2][2]), nt=nt)
                    b_nac_aft = Bezier(P=np.array(c.control_points[2][3]), nt=nt)
                b_hub_fore = Bezier(P=np.flipud(np.array(c.control_points[1][2])), nt=nt)
                b_hub_aft = Bezier(P=np.flipud(np.array(c.control_points[1][1])), nt=nt)
                c_nac_fore = np.column_stack((b_nac_fore.x, b_nac_fore.y))
                c_nac_aft = np.column_stack((b_nac_aft.x, b_nac_aft.y))[1:, :]
                c_nac = np.row_stack((c_nac_fore, c_nac_aft))
                c_hub_fore = np.column_stack((b_hub_fore.x, b_hub_fore.y))
                c_hub_aft = np.column_stack((b_hub_aft.x, b_hub_aft.y))[1:, :]
                c_hub = np.row_stack((c_hub_fore, c_hub_aft))
                c_nac_t = transformation.transform(c_nac)
                c_hub_t = transformation.transform(c_hub)
                radius = np.array([])
                for coord in c_nac_t:
                    r = abs(coord[1])
                    if coord[0] >= c_hub_t[0, 0]:
                        r -= abs(np.interp(x=np.array([coord[0]]), xp=c_hub_t[:, 0], fp=c_hub_t[:, 1])[0])
                    radius = np.append(radius, r)
                data = np.column_stack((c_nac_t[:, 0], radius))
                np.savetxt(os.path.join(self.analysis_dir, 'radius', f'radius_{idx}.dat'), data)
                np.savetxt(os.path.join(self.analysis_dir, 'radius', f'nac_{idx}.dat'), c_nac_t)
                np.savetxt(os.path.join(self.analysis_dir, 'radius', f'hub_{idx}.dat'), c_hub_t)

        if evaluate:
            population.eval_pop_fitness()
            population_forces = {k: [c.forces[k] for c in population.population]
                                 for k in population.population[0].forces.keys()}
            save_data(population_forces, self.post_process_force_file)

    def generate_aero_force_plot(self, var: str or list = None):
        post_process_forces = load_data(self.post_process_force_file)
        if isinstance(var, str):
            var = [var]
        if var is None:
            var = [k for k in post_process_forces.keys() if k not in ['converged', 'BL', 'errored_out', 'timed_out']]
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
        }
        for v in var:
            fig, axs = plt.subplots()
            if isinstance(post_process_forces['Cd'][0], typing.Iterable):
                axs.plot([el[1] * multiplier[v] for el in post_process_forces[v]], color=ILLINI_ORANGE)
            else:
                axs.plot([el * multiplier[v] for el in post_process_forces[v]], color=ILLINI_ORANGE)
            fig.set_tight_layout('tight')
            axs.set_xlabel("Generation", fontdict=font)
            axs.set_ylabel(ylabel[v], fontdict=font)
            axs.grid("on", ls=":")
            # save_name = os.path.join(self.image_dir, f'design_{v}.svg')
            # fig.savefig(save_name)
            save_name = os.path.join(self.image_dir, f'design_{v}.pdf')
            fig.savefig(save_name)

    def compare_geometries(self, index: list, plot_actuator_disk: bool = True):
        if len(index) != 2:
            raise ValueError(f"Comparison of only 2 geometries is supported. Current index list: {index}")
        fig, axs = plt.subplots(figsize=(12, 5))
        colors = [ILLINI_BLUE, ILLINI_ORANGE]
        save_filetypes = ['.svg', '.pdf']
        for analysis_idx, i in enumerate(index):
            coords = load_data(os.path.join(self.analysis_dir, 'coords', f'coords_{i}.json'))
            for airfoil in coords:
                airfoil = np.array(airfoil)
                axs.plot(airfoil[:, 0], airfoil[:, 1], color=colors[analysis_idx], ls='dashed')
            if plot_actuator_disk:
                control_points = load_data(os.path.join(self.analysis_dir,
                                                        'control_points', f'control_points_{i}.json'))
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
                axs.plot(ad1_x, ad1_y, color=colors[analysis_idx], ls='dotted')
                axs.plot(ad2_x, ad2_y, color=colors[analysis_idx], ls='dotted')

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
        axs.legend(legend_proxies, legend_names, prop={'size': 14, 'family': font['family']},
                   fancybox=True, shadow=True, loc=legend_loc, ncol=ncols)
        axs.set_aspect('equal')
        axs.set_xlabel(r'$x/c_{main}$', fontdict=font)
        axs.set_ylabel(r'$y/c_{main}$', fontdict=font)
        axs.grid('on', ls=':')
        for ext in save_filetypes:
            fig.savefig(os.path.join(self.image_dir, f"geometry_{index[0]}_{index[1]}{ext}"),
                        bbox_inches="tight")

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
            for analysis_idx, i in enumerate(index):
                bl_file = os.path.join(self.analysis_dir, 'analysis', f'analysis_{i}', f'bl.analysis_{i}')
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

                # Set the legend
                if mode == 'compare':
                    legend_proxies = [Line2D([], [], ls=ls, color=c)
                                      for ls, c in tuple(zip(line_styles * 2,
                                                             [colors[0], colors[0], colors[1], colors[1]]))]
                    fig.legend(legend_proxies, legend_strs, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 0.97),
                               ncol=len(legend_proxies), loc='upper center', prop={'size': 14,
                                                                                   'family': font['family']})

                if mode == 'standalone':
                    fig.savefig(os.path.join(self.image_dir, f'gen_{i}_{v}.svg'.replace('*', 'star')))
            if mode == 'compare':
                for ext in save_filetypes:
                    fig.savefig(os.path.join(self.image_dir,
                                             f'gen_{index[0]}_{index[1]}_{v}{ext}'.replace('*', 'star')),
                                bbox_inches="tight")

    def generate_field_matplotlib(self, var: str, axs: plt.Axes or None, index: int, cmap_field: mpl_colors.Colormap or str,
                                  cmap_airfoil: mpl_colors.Colormap or str, shading: str = 'gouraud', vmin: float = None,
                                  vmax: float = None):
        return generate_field_matplotlib(axs=axs,
                                         analysis_subdir=os.path.join(self.analysis_dir,
                                                                      'analysis', f'analysis_{index}'),
                                         var=var, cmap_field=cmap_field, cmap_airfoil=cmap_airfoil, shading=shading,
                                         vmin=vmin, vmax=vmax)

    def generate_single_field(self, var: str, index: int, cmap_field: mpl_colors.Colormap or str,
                              cmap_airfoil: mpl_colors.Colormap or str, shading: str = 'gouraud', vmin: float = None,
                              vmax: float = None, image_extensions: tuple = ('.png', '.pdf')):
        airfoil_color = 'black'
        flow_var_label_matplotlib = {'M': r'Mach Number',
                                     'Cp': r'Pressure Coefficient',
                                     'p': r'Static Pressure ($p/p_\infty$)',
                                     'rho': r'Density ($\rho/\rho_\infty$)',
                                     'u': r'Velocity-x ($u/V_\infty$)',
                                     'v': r'Velocity-y ($v/V_\infty$)',
                                     'q': r'Speed of Sound ($q/V_\infty$)'}
        post_process_forces = load_data(self.post_process_force_file)
        fig, axs = plt.subplots(figsize=(10, 5))
        # plt.subplots_adjust(left=0.08, bottom=-0.1, right=1.05, top=1.2, wspace=0.0, hspace=0.0)
        quad = generate_field_matplotlib(axs=axs,
                                  analysis_subdir=os.path.join(self.analysis_dir,
                                                               'analysis', f'analysis_{index}'),
                                  var=var, cmap_field=cmap_field, cmap_airfoil=cmap_airfoil, shading=shading,
                                  vmin=vmin, vmax=vmax)
        coords = load_data(os.path.join(self.analysis_dir, 'coords', f'coords_{index}.json'))
        for airfoil in coords:
            airfoil = np.array(airfoil)
            axs.plot(airfoil[:, 0], airfoil[:, 1], color=airfoil_color)
            polygon = Polygon(airfoil, closed=False, color="#000000AA")
            axs.add_patch(polygon)
        if not isinstance(post_process_forces['Cd'][index], typing.Iterable):
            cd_value = post_process_forces['Cd'][index]
        else:
            cd_value = post_process_forces['Cd'][index][1]
        cd_title = r'C_{F_{\parallel V_\infty}}'
        axs.text(x=-0.15, y=0.32, s=fr"Gen. {index}: ${cd_title} = {cd_value * 10000:.1f}$ counts",
                 fontdict=dict(size=18, family='serif'))
        cbar = fig.colorbar(quad[1]['field'], ax=axs, shrink=0.7)
        cbar.ax.set_ylabel(flow_var_label_matplotlib[var], fontdict=dict(size=18, family='serif'))
        cbar.ax.tick_params(axis='y', which='major', labelsize=14)
        axs.set_xlim([-0.2, 1.6])
        axs.set_ylim([-0.4, 0.4])
        axs.tick_params(axis='both', which='major', labelsize=14)
        axs.tick_params(axis='both', which='minor', labelsize=14)
        axs.set_aspect('equal')
        axs.set_xlabel(r"$x/c_{main}$", fontdict=dict(size=18, family='serif'))
        axs.set_ylabel(r"$y/c_{main}$", fontdict=dict(size=18, family='serif'))
        proxy_line = Line2D([], [], color=airfoil_color)
        proxy_patch_airfoil = Patch(color="#000000DD")
        proxy_patch_bl = Patch(color="#606060")
        if self.underwing:
            legend_loc = "upper right"
        else:
            legend_loc = "lower right"
        axs.legend((proxy_line, proxy_patch_airfoil, proxy_patch_bl), ('Surface', 'Solid', 'BL/Stag.'),
                   prop={'size': 17, 'family': font['family']}, loc=legend_loc)
        fig.set_tight_layout('tight')
        fig_fname_no_ext = os.path.join(self.image_dir, f'field_{index}_{var}')
        for ext in image_extensions:
            kwargs = {'bbox_inches': 'tight'}
            if ext == '.png':
                kwargs['dpi'] = 600
            fig.savefig(fig_fname_no_ext + ext, **kwargs)

    def generate_gif_from_images(self, var: str, index: int or typing.Iterable = None,
                                 save_name: str = 'opt_history.gif', duration: int = 150):
        """Borrowed from https://pythonprogramming.altervista.org/png-to-gif/"""
        index = self.set_index(index)
        frames = []
        for i in index:
            image = os.path.join(self.image_dir, f'field_{i}_{var}.png')
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
                                     'q': r'Speed of Sound ($q/V_\infty$)'}

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