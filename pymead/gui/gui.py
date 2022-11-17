from pymead.gui.rename_popup import RenamePopup
from pymead.gui.main_icon_toolbar import MainIconToolbar

from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, \
    QWidget, QMenu, QStatusBar, QAction, QLabel, QPushButton, QToolButton, QStyle, QTextEdit
from PyQt5.QtGui import QIcon, QFont, QFontDatabase, QPalette
from PyQt5.QtCore import QEvent, QObject, Qt, QPoint, QSize


from pymead.core.airfoil import Airfoil
from pymead import DATA_DIR, RESOURCE_DIR
from pymead.gui.input_dialog import SingleAirfoilViscousDialog, LoadDialog, SaveAsDialog, OptimizationSetupDialog
from pymead.gui.analysis_graph import AnalysisGraph
from pymead.gui.parameter_tree import MEAParamTree
from pymead.utils.airfoil_matching import match_airfoil
from pymead.analysis.single_element_inviscid import single_element_inviscid
from pymead.gui.text_area import ConsoleTextArea
from pymead.gui.dockable_tab_widget import DockableTabWidget
from pymead.core.mea import MEA
from pymead.analysis.calc_aero_data import calculate_aero_data
from pymead.optimization.opt_setup import CustomDisplay, TPAIOPT, SelfIntersectionRepair
from pymead.utils.read_write_files import load_data, save_data
from pymead.utils.misc import make_ga_opt_dir
from pymead.optimization.pop_chrom import Chromosome, Population, CustomGASettings
from pymead.optimization.custom_ga_sampling import CustomGASampling
from pymead.optimization.opt_setup import termination_condition, calculate_warm_start_index, \
    convert_opt_settings_to_param_dict
from pymead.gui.message_box import disp_message_box

import pymoo.core.population
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.config import Config
from pymoo.core.evaluator import Evaluator
from pymoo.factory import get_reference_directions
from pymoo.core.evaluator import set_cv

import pyqtgraph as pg
import numpy as np
import dill
from copy import deepcopy
from functools import partial
import sys
import os


class GUI(QMainWindow):
    def __init__(self, path=None, parent=None):
        # super().__init__(flags=Qt.FramelessWindowHint)
        super().__init__(parent=parent)
        # self.setWindowFlags(Qt.CustomizeWindowHint)
        self.path = path
        single_element_inviscid(np.array([[1, 0], [0, 0], [1, 0]]), 0.0)
        for font_name in ["DejaVuSans", "DejaVuSansMono", "DejaVuSerif"]:
            QFontDatabase.addApplicationFont(os.path.join(RESOURCE_DIR, "dejavu-fonts-ttf-2.37", "ttf",
                                                          f"{font_name}.ttf"))
        # QFontDatabase.addApplicationFont(os.path.join(RESOURCE_DIR, "cascadia-code", "Cascadia.ttf"))
        # print(QFontDatabase().families())

        self.design_tree = None
        self.dialog = None
        self.opt_settings = None
        self.analysis_graph = None
        self.te_thickness_edit_mode = False
        self.dark_mode = False
        self.n_analyses = 0
        self.n_converged_analyses = 0
        self.pens = [('#d4251c', Qt.SolidLine), ('darkorange', Qt.SolidLine), ('gold', Qt.SolidLine),
                     ('limegreen', Qt.SolidLine), ('cyan', Qt.SolidLine), ('mediumpurple', Qt.SolidLine),
                     ('deeppink', Qt.SolidLine), ('#d4251c', Qt.DashLine), ('darkorange', Qt.DashLine),
                     ('gold', Qt.DashLine),
                     ('limegreen', Qt.DashLine), ('cyan', Qt.DashLine), ('mediumpurple', Qt.DashLine),
                     ('deeppink', Qt.DashLine)]
        # self.setFont(QFont("DejaVu Serif"))
        self.setFont(QFont("DejaVu Sans"))

        self.mea = MEA(None, [Airfoil()], airfoil_graphs_active=True)
        # self.mea.airfoils['A0'].insert_free_point(FreePoint(Param(0.5), Param(0.1), previous_anchor_point='te_1'))
        # self.mea.airfoils['A0'].update()
        # self.airfoil_graphs = [AirfoilGraph(self.mea.airfoils['A0'])]
        self.w = self.mea.airfoils['A0'].airfoil_graph.w
        self.v = self.mea.airfoils['A0'].airfoil_graph.v
        # internal_geometry_xy = np.loadtxt(os.path.join(DATA_DIR, 'sec_6.txt'))
        # # print(f"geometry = {internal_geometry_xy}")
        # scale_factor = 0.612745
        # x_start = 0.13352022
        # self.internal_geometry = self.v.plot(internal_geometry_xy[:, 0] * scale_factor + x_start,
        #                                      internal_geometry_xy[:, 1] * scale_factor,
        #                                      pen=pg.mkPen(color='orange', width=1))
        # self.airfoil_graphs.append(AirfoilGraph(self.mea.airfoils['A1'], w=self.w, v=self.v))
        self.main_layout_upper = QVBoxLayout()
        self.main_layout = QHBoxLayout()
        self.setStatusBar(QStatusBar(self))
        self.param_tree_instance = MEAParamTree(self.mea, self.statusBar(), parent=self)
        self.mea.airfoils['A0'].airfoil_graph.param_tree = self.param_tree_instance
        self.mea.airfoils['A0'].airfoil_graph.airfoil_parameters = self.param_tree_instance.p.param(
            'Airfoil Parameters')
        # print(f"param_tree_instance = {self.param_tree_instance}")
        self.design_tree_widget = self.param_tree_instance.t
        # self.design_tree_widget.setAlternatingRowColors(False)
        # self.design_tree_widget.setStyleSheet("selection-background-color: #36bacfaa; selection-color: black;")
        # self.design_tree_widget.setStyleSheet('''QTreeWidget {color: black; alternate-background-color: red;
        #         selection-background-color: #36bacfaa;}
        #         QTreeView::item:hover {background: #36bacfaa;} QTreeView::item {border: 0px solid gray; color: black}''')
        self.setStyleSheet("color: black; font-family: DejaVu; font-size: 12px;")
        self.text_area = ConsoleTextArea()
        self.right_widget_layout = QVBoxLayout()
        # self.tab_widget = QTabWidget()
        # self.tab_widget.addTab(self.w, "Geometry")
        self.dockable_tab_window = DockableTabWidget(self)
        self.dockable_tab_window.add_new_tab_widget(self.w, "Geometry")

        self.right_widget_layout.addWidget(self.dockable_tab_window)
        self.right_widget_layout.addWidget(self.text_area)
        self.right_widget = QWidget()
        self.right_widget.setLayout(self.right_widget_layout)
        self.main_layout.addWidget(self.design_tree_widget, 1)
        self.main_layout.addWidget(self.right_widget, 3)
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        # print(f"children of gui = {self.main_widget.children()}")
        # self.airfoil_graph.w.setFocus()
        # self.main_layout_upper.addWidget(MyBar(self))
        # self.main_layout_upper.addWidget(self.main_widget)
        # self.main_widget_upper = QWidget()
        # self.main_widget_upper.setLayout(self.main_layout_upper)
        self.setCentralWidget(self.main_widget)

        # self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

        # self.resize(640, self.titleBar.height() + 480)

        self.set_title_and_icon()
        self.create_menu_bar()
        self.main_icon_toolbar = MainIconToolbar(self)
        if self.path is not None:
            self.load_mea_no_dialog(self.path)

    def set_dark_mode(self):
        self.setStyleSheet("background-color: #3e3f40; color: #dce1e6; font-family: DejaVu; font-size: 12px;")
        self.w.setBackground('#2a2a2b')

    def set_light_mode(self):
        self.setStyleSheet("font-family: DejaVu; font-size: 12px;")
        self.w.setBackground('w')

    def set_title_and_icon(self):
        self.setWindowTitle("pymead")
        image_path = os.path.join(os.path.dirname(os.getcwd()), 'icons', 'airfoil_slat.png')
        self.setWindowIcon(QIcon(image_path))

    def create_menu_bar(self):
        self.menu_bar = self.menuBar()
        # print(self.menu_bar)
        self.menu_names = {"&File": ["&Open", "&Save"]}
        # def recursively_add_menus(menu: dict, menu_bar: QMenu):
        #     for key, val in menu.items():
        #         if isinstance(val, dict):
        #             menu_bar.addMenu(QMenu(key, self))
        #             recursively_add_menus(val, menu_bar.children()[0])
        #         else:
        #

        # File Menu set-up
        self.file_menu = QMenu("&File", self)
        self.menu_bar.addMenu(self.file_menu)

        self.open_action = QAction("Open", self)
        self.file_menu.addAction(self.open_action)
        self.open_action.triggered.connect(self.load_mea)

        self.save_as_action = QAction("Save As", self)
        self.file_menu.addAction(self.save_as_action)
        self.save_as_action.triggered.connect(self.save_as_mea)

        self.save_action = QAction("Save", self)
        self.file_menu.addAction(self.save_action)
        self.save_action.triggered.connect(self.save_mea)

        self.settings_action = QAction("Settings", self)
        self.file_menu.addAction(self.settings_action)
        # self.settings_action.triggered.connect()

        # Analysis Menu set-up
        self.analysis_menu = QMenu("&Analysis", self)
        self.menu_bar.addMenu(self.analysis_menu)

        self.single_menu = QMenu("Single Airfoil", self)
        self.analysis_menu.addMenu(self.single_menu)
        self.multi_menu = QMenu("Multi-Element Airfoil", self)
        self.analysis_menu.addMenu(self.multi_menu)

        self.single_inviscid_action = QAction("Invisid", self)
        self.single_menu.addAction(self.single_inviscid_action)
        self.single_inviscid_action.triggered.connect(self.single_airfoil_inviscid_analysis)

        self.single_viscous_action = QAction("Viscous", self)
        self.single_menu.addAction(self.single_viscous_action)
        self.single_viscous_action.triggered.connect(self.single_airfoil_viscous_analysis)

        self.opt_menu = QMenu("&Optimization", self)
        self.menu_bar.addMenu(self.opt_menu)

        self.opt_run_action = QAction("Run", self)
        self.opt_menu.addAction(self.opt_run_action)
        self.opt_run_action.triggered.connect(self.run_optimization)

        self.tools_menu = QMenu("&Tools", self)
        self.menu_bar.addMenu(self.tools_menu)

        self.match_airfoil_action = QAction("Match Airfoil", self)
        self.tools_menu.addAction(self.match_airfoil_action)
        self.match_airfoil_action.triggered.connect(self.match_airfoil)

    def save_as_mea(self):
        dialog = SaveAsDialog(self)
        if dialog.exec_():
            self.mea.file_name = dialog.selectedFiles()[0]
            self.save_mea()
        else:
            pass

    def save_mea(self):
        with open(os.path.join(os.getcwd(), self.mea.file_name), "wb") as f:
            dill.dump(self.mea, f)
        for idx, airfoil in enumerate(self.mea.airfoils.values()):
            self.mea.add_airfoil_graph_to_airfoil(airfoil, idx, self.param_tree_instance, w=self.w, v=self.v)
        for a_name, a in self.mea.airfoils.items():
            a.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.param_tree_instance.plot_changed, a_name))

    def copy_mea(self):
        mea_copy = dill.copy(self.mea)
        for idx, airfoil in enumerate(self.mea.airfoils.values()):
            self.mea.add_airfoil_graph_to_airfoil(airfoil, idx, self.param_tree_instance, w=self.w, v=self.v)
        for a_name, a in self.mea.airfoils.items():
            a.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.param_tree_instance.plot_changed, a_name))
        return mea_copy

    def load_mea(self):
        dialog = LoadDialog(self)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
        else:
            file_name = None
        if file_name is not None:
            self.load_mea_no_dialog(file_name)

    def load_mea_no_dialog(self, file_name):
        with open(os.path.join(os.getcwd(), file_name), "rb") as f:
            self.mea = dill.load(f)
        self.v.clear()
        for idx, airfoil in enumerate(self.mea.airfoils.values()):
            self.mea.add_airfoil_graph_to_airfoil(airfoil, idx, None, w=self.w, v=self.v)
        self.param_tree_instance = MEAParamTree(self.mea, self.statusBar(), parent=self)
        for a in self.mea.airfoils.values():
            a.airfoil_graph.param_tree = self.param_tree_instance
            a.airfoil_graph.airfoil_parameters = a.airfoil_graph.param_tree.p.param('Airfoil Parameters')
        self.design_tree_widget = self.param_tree_instance.t
        self.main_layout.replaceWidget(self.main_layout.itemAt(0).widget(), self.design_tree_widget)
        self.v.autoRange()

    def disp_message_box(self, message: str, message_mode: str = 'error'):
        disp_message_box(message, self, message_mode=message_mode)

    def single_airfoil_inviscid_analysis(self):
        """Inviscid analysis not yet implemented here"""
        pass

    def single_airfoil_viscous_analysis(self):
        self.dialog = SingleAirfoilViscousDialog(items=[("Re", "double", 1e5), ("Iterations", "int", 150),
                                                        ("Timeout (seconds)", "double", 15),
                                                        ("Angle of Attack (degrees)", "double", 0.0),
                                                        ("Airfoil", "combo"), ("Name", "string", "default_airfoil"),
                                                        ("xtr upper", "double", 1.0), ("xtr lower", "double", 1.0),
                                                        ("Ncrit", "double", 9.0),
                                                        ("Use body-fixed CSYS", "checkbox", False)],
                                                 a_list=[k for k in self.mea.airfoils.keys()], parent=self)
        if self.dialog.exec():
            inputs = self.dialog.getInputs()
        else:
            inputs = None

        if inputs is not None:
            xfoil_settings = {'Re': inputs[0], 'timeout': inputs[2], 'iter': inputs[1], 'xtr': [inputs[6], inputs[7]],
                              'N': inputs[8]}
            aero_data, _ = calculate_aero_data(DATA_DIR, inputs[5], inputs[3], self.mea.airfoils[inputs[4]], 'xfoil',
                                               xfoil_settings, body_fixed_csys=inputs[9])
            if not aero_data['converged'] or aero_data['errored_out'] or aero_data['timed_out']:
                self.text_area.insertPlainText(
                    f"[{self.n_analyses:2.0f}] Converged = {aero_data['converged']} | Errored out = "
                    f"{aero_data['errored_out']} | Timed out = {aero_data['timed_out']}\n")
            else:
                self.text_area.insertPlainText(
                    f"[{self.n_analyses:2.0f}] {inputs[4]} (\u03b1 = {inputs[3]:5.2f} deg, Re = {inputs[0]:.3E}): "
                    f"Cl = {aero_data['Cl']:7.4f} | Cd = {aero_data['Cd']:.5f} (Cdp = {aero_data['Cdp']:.5f}, Cdf = {aero_data['Cdf']:.5f}) | Cm = {aero_data['Cm']:7.4f} "
                    f"| L/D = {aero_data['L/D']:8.4f}\n")
            sb = self.text_area.verticalScrollBar()
            sb.setValue(sb.maximum())

            if aero_data['converged'] and not aero_data['errored_out'] and not aero_data['timed_out']:
                if self.analysis_graph is None:
                    # Need to set analysis_graph to None if analysis window is closed! Might also not want to allow geometry docking window to be closed
                    if self.dark_mode:
                        bcolor = '#2a2a2b'
                    else:
                        bcolor = 'w'
                    self.analysis_graph = AnalysisGraph(background_color=bcolor)
                    self.dockable_tab_window.add_new_tab_widget(self.analysis_graph.w, "Analysis")
                pg_plot_handle = self.analysis_graph.v.plot(pen=pg.mkPen(color=self.pens[self.n_converged_analyses][0],
                                                                         style=self.pens[self.n_converged_analyses][1]),
                                                            name=str(self.n_analyses))
                pg_plot_handle.setData(aero_data['Cp']['x'], aero_data['Cp']['Cp'])
                # pen = pg.mkPen(color='green')
                self.n_converged_analyses += 1
                self.n_analyses += 1
            else:
                self.n_analyses += 1

    def match_airfoil(self):
        target_airfoil = 'A0'
        match_airfoil(self.mea, target_airfoil, 'sc20010-il')

    def run_optimization(self):
        exit_the_dialog = False
        early_return = False
        dialog = OptimizationSetupDialog(self)
        if dialog.exec_():
            while not exit_the_dialog and not early_return:
                self.opt_settings = dialog.getInputs()
                opt_settings = self.opt_settings
                Config.show_compile_hint = False

                param_dict = convert_opt_settings_to_param_dict(opt_settings)

                if opt_settings['Warm Start/Batch Mode']['use_current_mea']['state']:
                    mea = self.copy_mea()
                else:
                    mea_file = opt_settings['Warm Start/Batch Mode']['mea_file']['text']
                    if not os.path.exists(mea_file):
                        self.disp_message_box('MEAD parametrization file not found', message_mode='error')
                        exit_the_dialog = True
                        early_return = True
                        continue
                    else:
                        mea = load_data(mea_file)

                parameter_list = mea.extract_parameters()
                if isinstance(parameter_list, str):
                    error_message = parameter_list
                    self.disp_message_box(error_message, message_mode='error')
                    exit_the_dialog = True
                    early_return = True
                    continue

                param_dict['n_var'] = len(parameter_list)

                if opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
                    opt_dir = opt_settings['Warm Start/Batch Mode']['warm_start_dir']['text']
                else:
                    opt_dir = make_ga_opt_dir(opt_settings['Genetic Algorithm']['root_dir']['text'],
                                              opt_settings['Genetic Algorithm']['opt_dir_name']['text'])

                name_base = 'ga_airfoil'
                name = [f"{name_base}_{i}" for i in range(opt_settings['Genetic Algorithm']['n_offspring']['value'])]
                param_dict['name'] = name

                for airfoil in mea.airfoils.values():
                    airfoil.airfoil_graphs_active = False
                mea.airfoil_graphs_active = False
                base_folder = os.path.join(opt_settings['Genetic Algorithm']['root_dir']['text'],
                                           opt_settings['Genetic Algorithm']['temp_analysis_dir_name']['text'])
                param_dict['base_folder'] = base_folder

                if opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
                    param_dict['warm_start_generation'] = calculate_warm_start_index(
                        opt_settings['Warm Start/Batch Mode']['warm_start_generation']['value'], opt_dir)
                param_dict_save = deepcopy(param_dict)
                if not opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
                    save_data(param_dict_save, os.path.join(opt_dir, 'param_dict.json'))
                else:
                    save_data(param_dict_save, os.path.join(
                        opt_dir, f'param_dict_{param_dict["warm_start_generation"]}.json'))

                ref_dirs = get_reference_directions("energy", param_dict['n_obj'], param_dict['n_ref_dirs'],
                                                    seed=param_dict['seed'])
                ga_settings = CustomGASettings(population_size=param_dict['n_offsprings'],
                                               mutation_bounds=([-0.002, 0.002]),
                                               mutation_methods=('random-reset', 'random-perturb'),
                                               max_genes_to_mutate=2,
                                               mutation_probability=0.06,
                                               max_mutation_attempts_per_chromosome=500)

                problem = TPAIOPT(n_var=param_dict['n_var'], n_obj=param_dict['n_obj'], n_constr=param_dict['n_constr'],
                                  xl=param_dict['xl'], xu=param_dict['xu'], param_dict=param_dict, ga_settings=ga_settings)

                print(f"Made it here!!!")

                if not opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
                    tpaiga2_alg_instance = CustomGASampling(param_dict=problem.param_dict, ga_settings=ga_settings, mea=mea)
                    population = Population(problem.param_dict, ga_settings, generation=0,
                                            parents=[tpaiga2_alg_instance.generate_first_parent()],
                                            verbose=param_dict['verbose'], mea=mea)
                    population.generate()

                    n_subpopulations = 0
                    fully_converged_chromosomes = []
                    while True:  # "Do while" loop (terminate when enough of chromosomes have fully converged solutions)
                        subpopulation = deepcopy(population)
                        subpopulation.population = subpopulation.population[param_dict['num_processors'] * n_subpopulations:
                                                                            param_dict['num_processors'] * (
                                                                                    n_subpopulations + 1)]

                        subpopulation.eval_pop_fitness()

                        for chromosome in subpopulation.population:
                            if chromosome.fitness is not None:
                                fully_converged_chromosomes.append(chromosome)

                        if len(fully_converged_chromosomes) >= param_dict['population_size']:
                            # Truncate the list of fully converged chromosomes to just the first <population_size> number of
                            # chromosomes:
                            fully_converged_chromosomes = fully_converged_chromosomes[:param_dict['population_size']]
                            break

                        n_subpopulations += 1

                        if n_subpopulations * (param_dict['num_processors'] + 1) > param_dict['n_offsprings']:
                            raise Exception('Ran out of chromosomes to evaluate in initial population generation')

                    new_X = np.array([[]])
                    f1 = np.array([[]])
                    f2 = np.array([[]])

                    for chromosome in fully_converged_chromosomes:
                        if chromosome.fitness is not None:  # This statement should always pass, but shown here for clarity
                            if len(new_X) < 2:
                                new_X = np.append(new_X, np.array([chromosome.genes]))
                            else:
                                new_X = np.row_stack([new_X, np.array(chromosome.genes)])
                            f1_chromosome = np.array([chromosome.forces['Cd']])
                            f2_chromosome = np.array([np.abs(chromosome.forces['Cl'] - problem.target_CL)])
                            f1 = np.append(f1, f1_chromosome)
                            f2 = np.append(f2, f2_chromosome)

                            # write_F_X_data(1, chromosome, f1_chromosome[0], f2_chromosome[0],
                            #                force_and_obj_fun_file, design_variable_file, f_fmt, d_fmt)

                    pop_initial = pymoo.core.population.Population.new("X", new_X)
                    # objectives
                    pop_initial.set("F", np.column_stack([f1, f2]))
                    # set_cv(pop_initial)
                    Evaluator(skip_already_evaluated=True).eval(problem, pop_initial)

                    algorithm = UNSGA3(ref_dirs=ref_dirs, sampling=pop_initial, repair=SelfIntersectionRepair(mea=mea),
                                       n_offsprings=param_dict['n_offsprings'],
                                       crossover=SimulatedBinaryCrossover(eta=param_dict['eta_crossover']),
                                       mutation=PolynomialMutation(eta=param_dict['eta_mutation']))

                    termination = termination_condition(param_dict)

                    # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
                    algorithm.setup(problem, termination, display=CustomDisplay(), seed=param_dict['seed'], verbose=True,
                                    save_history=True)

                    save_data(algorithm, os.path.join(opt_dir, 'algorithm_gen_0.pkl'))

                    # np.save('checkpoint', algorithm)
                    # until the algorithm has no terminated
                    n_generation = 0
                else:
                    warm_start_index = param_dict['warm_start_generation']
                    n_generation = warm_start_index
                    algorithm = load_data(os.path.join(opt_settings['Warm Start/Batch Mode']['warm_start_dir']['text'],
                                                       f'algorithm_gen_{warm_start_index}.pkl'))
                    term = deepcopy(algorithm.termination.terminations)
                    term = list(term)
                    term[0].n_max_gen = param_dict['n_max_gen']
                    term = tuple(term)
                    algorithm.termination.terminations = term
                    algorithm.has_terminated = False

                while algorithm.has_next():

                    pop = algorithm.ask()

                    n_generation += 1

                    if n_generation > 1:

                        # evaluate (objective function value arrays must be numpy column vectors)
                        X = pop.get("X")
                        new_X = np.array([[]])
                        f1 = np.array([[]])
                        f2 = np.array([[]])
                        n_infeasible_solutions = 0
                        search_for_feasible_idx = 0
                        while True:
                            gene_matrix = []
                            feasible_indices = []
                            while True:
                                if X[search_for_feasible_idx, 0] != 9999:
                                    gene_matrix.append(X[search_for_feasible_idx, :].tolist())
                                    feasible_indices.append(search_for_feasible_idx)
                                else:
                                    n_infeasible_solutions += 1
                                search_for_feasible_idx += 1
                                if len(gene_matrix) == problem.num_processors:
                                    break
                            population = [Chromosome(problem.param_dict, ga_settings=ga_settings, category=None,
                                                     generation=n_generation,
                                                     population_idx=feasible_indices[idx + len(feasible_indices)
                                                                                     - param_dict['num_processors']],
                                                     genes=gene_list, verbose=param_dict['verbose'],
                                                     mea=mea)
                                          for idx, gene_list in enumerate(gene_matrix)]
                            pop_obj = Population(problem.param_dict, ga_settings=ga_settings, generation=n_generation,
                                                 parents=population, verbose=param_dict['verbose'], mea=mea)
                            pop_obj.population = population
                            for chromosome in pop_obj.population:
                                chromosome.generate()
                            pop_obj.eval_pop_fitness()
                            for idx, chromosome in enumerate(pop_obj.population):
                                if chromosome.fitness is not None:
                                    if len(new_X) < 2:
                                        new_X = np.append(new_X, np.array([chromosome.genes]))
                                    else:
                                        new_X = np.row_stack([new_X, np.array(chromosome.genes)])
                                    f1_chromosome = np.array([1.0 * chromosome.forces['Cd']])
                                    f2_chromosome = np.array([np.abs(chromosome.forces['Cl'] - problem.target_CL)])
                                    f1 = np.append(f1, f1_chromosome)
                                    f2 = np.append(f2, f2_chromosome)

                                else:
                                    if len(new_X) < 2:
                                        new_X = np.append(new_X, np.array([chromosome.genes]))
                                    else:
                                        new_X = np.row_stack([new_X, np.array(chromosome.genes)])
                                    f1 = np.append(f1, np.array([1000.0]))
                                    f2 = np.append(f2, np.array([1000.0]))
                            algorithm.evaluator.n_eval += problem.num_processors
                            population_full = (f1 < 1000.0).sum() >= param_dict['population_size']
                            if population_full:
                                break
                        # Set the objective function values of the remaining individuals to 1000.0
                        for idx in range(search_for_feasible_idx, len(X)):
                            new_X = np.row_stack([new_X, X[idx, :]])
                            f1 = np.append(f1, np.array([1000.0]))
                            f2 = np.append(f2, np.array([1000.0]))
                        new_X = np.append(new_X, 9999 * np.ones(shape=(n_infeasible_solutions, param_dict['n_var'])),
                                          axis=0)
                        for idx in range(n_infeasible_solutions):
                            f1 = np.append(f1, np.array([1000.0]))
                            f2 = np.append(f2, np.array([1000.0]))

                        pop.set("X", new_X)

                        # objectives
                        pop.set("F", np.column_stack([f1, f2]))

                        # for constraints
                        # pop.set("G", the_constraint_values))

                        # this line is necessary to set the CV and feasbility status - even for unconstrained
                        set_cv(pop)

                    # returned the evaluated individuals which have been evaluated or even modified
                    algorithm.tell(infills=pop)

                    # do same more things, printing, logging, storing or even modifying the algorithm object
                    if n_generation % param_dict['algorithm_save_frequency'] == 0:
                        save_data(algorithm, os.path.join(opt_dir, f'algorithm_gen_{n_generation}.pkl'))

                # obtain the result objective from the algorithm
                res = algorithm.result()
                save_data(res, os.path.join(opt_dir, 'res.pkl'))
                exit_the_dialog = True
        else:
            return
        if early_return:
            self.run_optimization()

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.ContextMenu and source is self.design_tree:
            menu = QMenu()
            menu.addAction('Rename')

            if menu.exec_(event.globalPos()):
                item = source.itemAt(event.pos())
                if item.text(0) not in ['Airfoils', 'Curves']:
                    rename_popup = RenamePopup(item.text(0), item)
                    rename_popup.exec()
            return True

        return super().eventFilter(source, event)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    if len(sys.argv) > 1:
        gui = GUI(sys.argv[1])
    else:
        gui = GUI()
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
