from abc import ABCMeta, abstractmethod
from PyQt5.QtCore import QObject
from pymead.core.mea import MEA
from pymead.gui.opt_airfoil_graph import OptAirfoilGraph
from pymead.gui.parallel_coords_graph import ParallelCoordsGraph
from pymead.gui.aero_forces_graphs import DragGraph, CpGraph
import pyqtgraph as pg
import numpy as np


class OptCallback(QObject):
    def __init__(self, parent):
        super().__init__(parent=parent)

    @abstractmethod
    def exec_callback(self):
        return


class PlotAirfoilCallback(OptCallback):
    def __init__(self, parent, mea: MEA, X, background_color: str = 'w'):
        super().__init__(parent=parent)
        self.mea = mea
        self.X = X
        self.parent = parent
        self.background_color = background_color

    def exec_callback(self):
        self.mea.update_parameters(self.X)
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
        pg_plot_handle = self.parent.opt_airfoil_graph.v.plot(pen=pen)
        coords = self.mea.airfoils['A0'].get_coords(body_fixed_csys=False)
        # print(f"thickness = {self.mea.airfoils['A0'].compute_thickness()[2]}")
        pg_plot_handle.setData(coords[:, 0], coords[:, 1])
        if len(self.parent.opt_airfoil_plot_handles) > 1:
            self.parent.opt_airfoil_plot_handles[-1].setPen(old_pen)
        self.parent.opt_airfoil_plot_handles.append(pg_plot_handle)


class ParallelCoordsCallback(OptCallback):
    def __init__(self, parent, mea: MEA, X, background_color: str = 'w'):
        super().__init__(parent=parent)
        self.mea = mea
        self.X = X
        self.parent = parent
        self.background_color = background_color

    def exec_callback(self):
        self.mea.update_parameters(self.X)
        norm_val_list, param_list = self.mea.extract_parameters(write_to_txt_file=False)
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
    def __init__(self, parent, background_color: str = 'w'):
        super().__init__(parent=parent)
        self.parent = parent
        self.background_color = background_color
        self.forces = self.parent.forces_dict
        self.Cd = self.forces['Cd']
        self.Cdp = self.forces['Cdp']
        self.Cdf = self.forces['Cdf']

    def exec_callback(self):
        if self.parent.drag_graph is None:
            self.parent.drag_graph = DragGraph(background_color=self.background_color)
        tab_name = "Drag"
        if tab_name not in self.parent.dockable_tab_window.names:
            self.parent.dockable_tab_window.add_new_tab_widget(self.parent.drag_graph.w, tab_name)
        self.parent.drag_graph.pg_plot_handle_Cd.setData(1e4 * np.array(self.Cd))
        self.parent.drag_graph.pg_plot_handle_Cdp.setData(1e4 * np.array(self.Cdp))
        self.parent.drag_graph.pg_plot_handle_Cdf.setData(1e4 * np.array(self.Cdf))


class DragPlotCallbackMSES(OptCallback):
    def __init__(self, parent, background_color: str = 'w'):
        super().__init__(parent=parent)
        self.parent = parent
        self.background_color = background_color
        self.forces = self.parent.forces_dict
        self.Cd = self.forces['Cd']
        # self.Cdp = self.forces['Cdp']
        # self.Cdf = self.forces['Cdf']

    def exec_callback(self):
        if self.parent.drag_graph is None:
            self.parent.drag_graph = DragGraph(background_color=self.background_color)
        tab_name = "Drag"
        if tab_name not in self.parent.dockable_tab_window.names:
            self.parent.dockable_tab_window.add_new_tab_widget(self.parent.drag_graph.w, tab_name)
        self.parent.drag_graph.pg_plot_handle_Cd.setData(1e4 * np.array(self.Cd))
        # self.parent.drag_graph.pg_plot_handle_Cdp.setData(1e4 * np.array(self.Cdp))
        # self.parent.drag_graph.pg_plot_handle_Cdf.setData(1e4 * np.array(self.Cdf))


class CpPlotCallbackXFOIL(OptCallback):
    def __init__(self, parent, background_color: str = 'w'):
        super().__init__(parent=parent)
        self.parent = parent
        self.background_color = background_color
        self.forces = self.parent.forces_dict
        self.Cp = self.forces['Cp'][-1]

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
        pg_plot_handle.setData(self.Cp['x'], self.Cp['Cp'])
        if len(self.parent.Cp_graph_plot_handles) > 1:
            self.parent.Cp_graph_plot_handles[-1].setPen(old_pen)
        self.parent.Cp_graph_plot_handles.append(pg_plot_handle)


class CpPlotCallbackMSES(OptCallback):
    def __init__(self, parent, background_color: str = 'w'):
        super().__init__(parent=parent)
        self.parent = parent
        self.background_color = background_color
        self.forces = self.parent.forces_dict
        # print(f"forces = {self.forces}")
        self.Cp = self.forces['BL'][-1]

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
        pg_plot_handle.setData(self.Cp[0]['x'], self.Cp[0]['Cp'])
        if len(self.parent.Cp_graph_plot_handles) > 1:
            self.parent.Cp_graph_plot_handles[-1].setPen(old_pen)
        self.parent.Cp_graph_plot_handles.append(pg_plot_handle)
