import typing
from abc import abstractmethod
from PyQt5.QtCore import QObject
from PyQt5.QtGui import QFontDatabase

from pymead.core.mea import MEA
from pymead.gui.opt_airfoil_graph import OptAirfoilGraph
from pymead.gui.parallel_coords_graph import ParallelCoordsGraph
from pymead.gui.aero_forces_graphs import DragGraph, CpGraph
import pyqtgraph as pg
import numpy as np
from copy import deepcopy


class OptCallback(QObject):
    def __init__(self, parent):
        super().__init__(parent=parent)

    @abstractmethod
    def exec_callback(self):
        return


class TextCallback(OptCallback):
    def __init__(self, parent, text_list: typing.List[list], completed: bool = False):
        super().__init__(parent=parent)
        self.parent = parent
        self.text_list = text_list
        self.names = [attr[0] for attr in self.text_list]
        self.widths = [attr[2] for attr in self.text_list]
        self.values = [attr[1] for attr in self.text_list]
        self.completed = completed

    def generate_header(self):
        t = ""
        for idx, (w, n) in enumerate(zip(self.widths, self.names)):
            if idx == 0:
                t += "||"
            t += f"{n.center(w + 2)}|"
            if idx == len(self.widths) - 1:
                t += "|"
        header_length = len(t)
        return "="*header_length + "\n" + t + "\n" + "="*header_length

    def stringify_text_list(self):
        t = ""
        for idx, (w, v) in enumerate(zip(self.widths, self.values)):
            if idx == 0:
                t += "||"
            if not isinstance(v, str):
                v = str(v)
            t += f"{v.center(w + 2)}|"
            if idx == len(self.widths) - 1:
                t += "|"
        print(f"Row length = {len(t)}")
        return t

    @staticmethod
    def generate_closer(length: int):
        return "="*length

    def exec_callback(self):
        if self.values[0] == 1:
            # font = self.parent.text_area.font()
            # print(QFontDatabase().families())
            # # font.setFamily("DejaVu Sans Mono")
            # font.setFamily("Courier")
            # font.setPointSize(10)
            # self.parent.text_area.setFont(font)
            # self.parent.output_area_text(f"<head><style>body {{font-family: DejaVu Sans Mono;}}</style></head><body><p><font size='4'>&#8203;</font></p></body>", mode="html")
            # self.parent.output_area_text("\n")
            self.parent.output_area_text(f"{self.generate_header()}")
            self.parent.output_area_text("\n")
        t = f"{self.stringify_text_list()}"
        self.parent.output_area_text(t)
        self.parent.output_area_text("\n")
        if self.completed:
            print("Completed!")
            self.parent.output_area_text(f"{self.generate_closer(len(t))}")
            self.parent.output_area_text("\n")


class PlotAirfoilCallback(OptCallback):
    def __init__(self, parent, mea: dict, X, background_color: str = 'w'):
        super().__init__(parent=parent)
        self.mea = deepcopy(mea)
        self.mea_object = MEA.generate_from_param_dict(self.mea)
        self.X = X
        self.parent = parent
        self.background_color = background_color

    def exec_callback(self):
        self.mea_object.update_parameters(self.X)
        if self.parent.opt_airfoil_graph is None:
            self.parent.opt_airfoil_graph = OptAirfoilGraph(background_color=self.background_color)
        tab_name = "Opt. Airfoil"
        if tab_name not in self.parent.dockable_tab_window.names:
            self.parent.dockable_tab_window.add_new_tab_widget(self.parent.opt_airfoil_graph.w, tab_name)
        if len(self.parent.opt_airfoil_plot_handles) > 0:
            pen = pg.mkPen(color='limegreen', width=2)
        else:
            pen = pg.mkPen(color='lightcoral', width=2)
        old_pen = pg.mkPen(color=pg.mkColor('l'), width=1)
        pg_plot_handles = []
        temp_output_airfoil = None
        for airfoil in self.mea_object.airfoils.values():
            pg_plot_handle = self.parent.opt_airfoil_graph.v.plot(pen=pen)
            coords = airfoil.get_coords(body_fixed_csys=False)
            # print(f"thickness = {self.mea.airfoils['A0'].compute_thickness()[2]}")
            pg_plot_handle.setData(coords[:, 0], coords[:, 1])
            pg_plot_handles.append(pg_plot_handle)
        if len(self.parent.opt_airfoil_plot_handles) > 1:
            for plot_handle in self.parent.opt_airfoil_plot_handles[-1]:
                plot_handle.setPen(old_pen)
        self.parent.opt_airfoil_plot_handles.append(pg_plot_handles)


class ParallelCoordsCallback(OptCallback):
    def __init__(self, parent, mea: dict, X, background_color: str = 'w'):
        super().__init__(parent=parent)
        self.mea = deepcopy(mea)
        self.mea_object = MEA.generate_from_param_dict(self.mea)
        self.X = X
        self.parent = parent
        self.background_color = background_color

    def exec_callback(self):
        self.mea_object.update_parameters(self.X)
        norm_val_list, param_list = self.mea_object.extract_parameters()
        if self.parent.parallel_coords_graph is None:
            self.parent.parallel_coords_graph = ParallelCoordsGraph(background_color=self.background_color)
        tab_name = "Parallel Coordinates"
        if tab_name not in self.parent.dockable_tab_window.names:
            self.parent.dockable_tab_window.add_new_tab_widget(self.parent.parallel_coords_graph.w, tab_name)
        if len(self.parent.parallel_coords_plot_handles) > 0:
            pen = pg.mkPen(color='limegreen', width=2)
        else:
            pen = pg.mkPen(color='lightcoral', width=2)
        old_pen = pg.mkPen(color=pg.mkColor('l'), width=1)
        if len(self.parent.parallel_coords_plot_handles) == 0:
            parameter_name_list = [param.name for param in param_list]
            xdict = dict(enumerate(parameter_name_list))
            stringaxis = pg.AxisItem(orientation='bottom')
            stringaxis.setTicks([xdict.items()])
            self.parent.parallel_coords_graph.v.setAxisItems({'bottom': stringaxis})
        pg_plot_handle = self.parent.parallel_coords_graph.v.plot(pen=pen)
        pg_plot_handle.setData(norm_val_list)
        if len(self.parent.parallel_coords_plot_handles) > 1:
            self.parent.parallel_coords_plot_handles[-1].setPen(old_pen)
        self.parent.parallel_coords_plot_handles.append(pg_plot_handle)


class DragPlotCallbackXFOIL(OptCallback):
    def __init__(self, parent, background_color: str = 'w', design_idx: int = 0):
        super().__init__(parent=parent)
        self.parent = parent
        self.background_color = background_color
        self.forces = self.parent.forces_dict
        self.Cd = self.forces['Cd']
        self.Cdp = self.forces['Cdp']
        self.Cdf = self.forces['Cdf']
        print(f"{self.Cd = }")
        self.design_idx = design_idx

    def exec_callback(self):
        if self.parent.drag_graph is None:
            self.parent.drag_graph = DragGraph(background_color=self.background_color)
        tab_name = "Drag"
        if tab_name not in self.parent.dockable_tab_window.names:
            self.parent.dockable_tab_window.add_new_tab_widget(self.parent.drag_graph.w, tab_name)
        Cd = self.Cd if not isinstance(self.Cd[0], list) else [d[self.design_idx] for d in self.Cd]
        Cdp = self.Cdp if not isinstance(self.Cdp[0], list) else [d[self.design_idx] for d in self.Cdp]
        Cdf = self.Cdf if not isinstance(self.Cdf[0], list) else [d[self.design_idx] for d in self.Cdf]
        self.parent.drag_graph.pg_plot_handle_Cd.setData(1e4 * np.array(Cd))
        self.parent.drag_graph.pg_plot_handle_Cdp.setData(1e4 * np.array(Cdp))
        self.parent.drag_graph.pg_plot_handle_Cdf.setData(1e4 * np.array(Cdf))


class DragPlotCallbackMSES(OptCallback):
    def __init__(self, parent, background_color: str = 'w', design_idx: int = 0):
        super().__init__(parent=parent)
        self.parent = parent
        self.background_color = background_color
        self.forces = self.parent.forces_dict
        self.Cd = self.forces['Cd']
        self.Cdp = self.forces['Cdp']
        self.Cdf = self.forces['Cdf']
        self.Cdv = self.forces['Cdv']
        self.Cdw = self.forces['Cdw']
        self.design_idx = design_idx

    def exec_callback(self):
        if self.parent.drag_graph is None:
            self.parent.drag_graph = DragGraph(background_color=self.background_color)
        tab_name = "Drag"
        if tab_name not in self.parent.dockable_tab_window.names:
            self.parent.dockable_tab_window.add_new_tab_widget(self.parent.drag_graph.w, tab_name)
        Cd = self.Cd if not isinstance(self.Cd[0], list) else [d[self.design_idx] for d in self.Cd]
        Cdp = self.Cdp if not isinstance(self.Cdp[0], list) else [d[self.design_idx] for d in self.Cdp]
        Cdf = self.Cdf if not isinstance(self.Cdf[0], list) else [d[self.design_idx] for d in self.Cdf]
        Cdv = self.Cdv if not isinstance(self.Cdv[0], list) else [d[self.design_idx] for d in self.Cdv]
        Cdw = self.Cdw if not isinstance(self.Cdw[0], list) else [d[self.design_idx] for d in self.Cdw]
        self.parent.drag_graph.pg_plot_handle_Cd.setData(1e4 * np.array(Cd))
        self.parent.drag_graph.pg_plot_handle_Cdp.setData(1e4 * np.array(Cdp))
        self.parent.drag_graph.pg_plot_handle_Cdf.setData(1e4 * np.array(Cdf))
        self.parent.drag_graph.pg_plot_handle_Cdv.setData(1e4 * np.array(Cdv))
        self.parent.drag_graph.pg_plot_handle_Cdw.setData(1e4 * np.array(Cdw))


class CpPlotCallbackXFOIL(OptCallback):
    def __init__(self, parent, background_color: str = 'w', design_idx: int = 0):
        super().__init__(parent=parent)
        self.parent = parent
        self.background_color = background_color
        self.forces = self.parent.forces_dict
        self.Cp = self.forces['Cp'][-1]
        self.design_idx = design_idx

    def exec_callback(self):
        if self.parent.Cp_graph is None:
            self.parent.Cp_graph = CpGraph(background_color=self.background_color)
        tab_name = "Cp"
        if tab_name not in self.parent.dockable_tab_window.names:
            self.parent.dockable_tab_window.add_new_tab_widget(self.parent.Cp_graph.w, tab_name)
        if len(self.parent.Cp_graph_plot_handles) > 0:
            pen = pg.mkPen(color='limegreen', width=2)
        else:
            pen = pg.mkPen(color='lightcoral', width=2)
        old_pen = pg.mkPen(color=pg.mkColor('l'), width=1)
        pg_plot_handle = self.parent.Cp_graph.v.plot(pen=pen)
        Cp = self.Cp if not isinstance(self.Cp, list) else self.Cp[self.design_idx]
        pg_plot_handle.setData(Cp['x'], Cp['Cp'])
        if len(self.parent.Cp_graph_plot_handles) > 1:
            self.parent.Cp_graph_plot_handles[-1].setPen(old_pen)
        self.parent.Cp_graph_plot_handles.append(pg_plot_handle)


class CpPlotCallbackMSES(OptCallback):
    def __init__(self, parent, background_color: str = 'w', design_idx: int = 0):
        super().__init__(parent=parent)
        self.parent = parent
        self.background_color = background_color
        self.forces = self.parent.forces_dict
        # print(f"forces = {self.forces}")
        self.Cp = self.forces['BL'][-1] if isinstance(
            self.forces['BL'][-1][0], dict) else self.forces['BL'][-1][design_idx]
        self.x_max = self.parent.mea.calculate_max_x_extent()

    def exec_callback(self):
        if self.parent.Cp_graph is None:
            self.parent.Cp_graph = CpGraph(background_color=self.background_color)
        tab_name = "Cp"
        if tab_name not in self.parent.dockable_tab_window.names:
            self.parent.dockable_tab_window.add_new_tab_widget(self.parent.Cp_graph.w, tab_name)
        if len(self.parent.Cp_graph_plot_handles) > 0:
            pen = pg.mkPen(color='limegreen', width=2)
        else:
            pen = pg.mkPen(color='lightcoral', width=2)
        old_pen = pg.mkPen(color=pg.mkColor('l'), width=1)
        pg_plot_handles = [self.parent.Cp_graph.v.plot(pen=pen) for _ in range(len(self.Cp))]
        for idx, side in enumerate(self.Cp):
            x = side['x']
            Cp = side['Cp']
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            if not isinstance(Cp, np.ndarray):
                Cp = np.array(Cp)
            pg_plot_handles[idx].setData(x[np.where(x <= self.x_max)[0]], Cp[np.where(x <= self.x_max)[0]])
        if len(self.parent.Cp_graph_plot_handles) > 1:
            for handle in self.parent.Cp_graph_plot_handles[-1]:
                handle.setPen(old_pen)
        self.parent.Cp_graph_plot_handles.append(pg_plot_handles)
