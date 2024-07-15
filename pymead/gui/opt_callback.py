import typing
from abc import abstractmethod

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject

from pymead.core.geometry_collection import GeometryCollection
from pymead.gui.analysis_graph import OptAirfoilGraph, ParallelCoordsGraph, DragGraph, CpGraph
from pymead.gui.vertical_axis_item import AngledAxisItem


class OptCallback(QObject):
    def __init__(self, parent):
        super().__init__(parent=parent)

    @abstractmethod
    def exec_callback(self, *args, **kwargs):
        return


class TextCallback(OptCallback):
    def __init__(self, parent, text_list: typing.List[list], completed: bool = False,
                 warm_start_gen: int or None = None):
        super().__init__(parent=parent)
        self.parent = parent
        self.text_list = text_list
        self.names = [attr[0] for attr in self.text_list]
        self.widths = [attr[2] for attr in self.text_list]
        self.values = [attr[1] for attr in self.text_list]
        self.completed = completed
        self.warm_start_gen = warm_start_gen

    def generate_header(self):
        t = ""
        for idx, (w, n) in enumerate(zip(self.widths, self.names)):
            if idx == 0:
                t += "||"
            t += f"{n.center(w + 2)}|"
            if idx == len(self.widths) - 1:
                t += "|"
        header_length = len(t)
        return ["="*header_length, t, "="*header_length]

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
        # print(f"Row length = {len(t)}")
        return t

    @staticmethod
    def generate_closer(length: int):
        return "="*length

    def exec_callback(self):
        if self.values[0] == 1 or (self.warm_start_gen is not None and self.values[0] == self.warm_start_gen + 1):
            # font = self.parent.text_area.font()
            # print(QFontDatabase().families())
            # # font.setFamily("DejaVu Sans Mono")
            # font.setFamily("Courier")
            # font.setPointSize(10)
            # self.parent.text_area.setFont(font)
            # self.parent.output_area_text(f"<head><style>body {{font-family: DejaVu Sans Mono;}}</style></head><body><p><font size='4'>&#8203;</font></p></body>", mode="html")
            # self.parent.output_area_text("\n")
            for header_line in self.generate_header():
                self.parent.output_area_text(header_line, line_break=True)
            # self.parent.output_area_text("\n")
        t = f"{self.stringify_text_list()}"
        self.parent.output_area_text(t, line_break=True)
        self.parent.closer = self.generate_closer(len(t))
        if self.completed:
            self.parent.output_area_text(f"{self.generate_closer(len(t))}", line_break=True)


class PlotAirfoilCallback(OptCallback):
    def __init__(self, parent, coords: list, geo_col: GeometryCollection):
        super().__init__(parent=parent)
        self.coords = coords
        self.parent = parent
        self.geo_col = geo_col

    def exec_callback(self, theme: dict, grid: bool):
        if self.parent.opt_airfoil_graph is None:
            self.parent.opt_airfoil_graph = OptAirfoilGraph(theme, geo_col=self.geo_col, grid=grid)
        tab_name = "Opt. Airfoil"
        if tab_name not in self.parent.dock_widget_names:
            self.parent.add_new_tab_widget(self.parent.opt_airfoil_graph.w, tab_name)
        if len(self.parent.opt_airfoil_plot_handles) > 0:
            pen = pg.mkPen(color='limegreen', width=2)
        else:
            pen = pg.mkPen(color='lightcoral', width=2)
        old_pen = pg.mkPen(color=pg.mkColor('l'), width=1)
        pg_plot_handles = []
        temp_output_airfoil = None
        for coords in self.coords:
            pg_plot_handle = self.parent.opt_airfoil_graph.v.plot(pen=pen)
            # print(f"thickness = {self.mea.airfoils['A0'].compute_thickness()[2]}")
            pg_plot_handle.setData(coords[:, 0], coords[:, 1])
            pg_plot_handles.append(pg_plot_handle)
        if len(self.parent.opt_airfoil_plot_handles) > 1:
            for plot_handle in self.parent.opt_airfoil_plot_handles[-1]:
                plot_handle.setPen(old_pen)
        self.parent.opt_airfoil_plot_handles.append(pg_plot_handles)


class ParallelCoordsCallback(OptCallback):
    def __init__(self, parent, norm_val_list: list, param_name_list: list):
        super().__init__(parent=parent)
        self.norm_val_list = norm_val_list
        self.param_name_list = param_name_list
        self.parent = parent

    def exec_callback(self, theme: dict, grid: bool):
        if self.parent.parallel_coords_graph is None:
            self.parent.parallel_coords_graph = ParallelCoordsGraph(theme, grid=grid)
        tab_name = "Parallel Coordinates"
        if tab_name not in self.parent.dock_widget_names:
            self.parent.add_new_tab_widget(self.parent.parallel_coords_graph.w, tab_name)
        if len(self.parent.parallel_coords_plot_handles) > 0:
            pen = pg.mkPen(color='limegreen', width=2)
        else:
            pen = pg.mkPen(color='lightcoral', width=2)
        old_pen = pg.mkPen(color=pg.mkColor('l'), width=1)
        if len(self.parent.parallel_coords_plot_handles) == 0:
            xdict = dict(enumerate(self.param_name_list))
            stringaxis = AngledAxisItem(orientation='bottom', angle=90)
            stringaxis.setTicks([xdict.items()])
            self.parent.parallel_coords_graph.v.setAxisItems({'bottom': stringaxis})
            self.parent.parallel_coords_graph.set_formatting(theme)
        pg_plot_handle = self.parent.parallel_coords_graph.v.plot(pen=pen)
        pg_plot_handle.setData(self.norm_val_list)
        if len(self.parent.parallel_coords_plot_handles) > 1:
            self.parent.parallel_coords_plot_handles[-1].setPen(old_pen)
        self.parent.parallel_coords_plot_handles.append(pg_plot_handle)


class DragPlotCallbackXFOIL(OptCallback):
    def __init__(self, parent, Cd, Cdp, Cdf):
        super().__init__(parent=parent)
        self.parent = parent
        self.Cd = Cd
        self.Cdp = Cdp
        self.Cdf = Cdf

    def exec_callback(self, theme: dict, grid: bool):
        if self.parent.drag_graph is None:
            self.parent.drag_graph = DragGraph(theme, grid=grid)
        tab_name = "Drag"
        if tab_name not in self.parent.dock_widget_names:
            self.parent.add_new_tab_widget(self.parent.drag_graph.w, tab_name)
        self.parent.drag_graph.pg_plot_handle_Cd.setData(1e4 * np.array(self.Cd))
        self.parent.drag_graph.pg_plot_handle_Cdp.setData(1e4 * np.array(self.Cdp))
        self.parent.drag_graph.pg_plot_handle_Cdf.setData(1e4 * np.array(self.Cdf))


class DragPlotCallbackMSES(OptCallback):
    def __init__(self, parent, Cd, Cdp, Cdf, Cdv, Cdw):
        super().__init__(parent=parent)
        self.parent = parent
        self.Cd = Cd
        self.Cdp = Cdp
        self.Cdf = Cdf
        self.Cdv = Cdv
        self.Cdw = Cdw

    def exec_callback(self, theme: dict, grid: bool):
        if self.parent.drag_graph is None:
            self.parent.drag_graph = DragGraph(theme, grid=grid)
        tab_name = "Drag"
        if tab_name not in self.parent.dock_widget_names:
            self.parent.add_new_tab_widget(self.parent.drag_graph.w, tab_name)
        self.parent.drag_graph.pg_plot_handle_Cd.setData(1e4 * np.array(self.Cd))
        self.parent.drag_graph.pg_plot_handle_Cdp.setData(1e4 * np.array(self.Cdp))
        self.parent.drag_graph.pg_plot_handle_Cdf.setData(1e4 * np.array(self.Cdf))
        self.parent.drag_graph.pg_plot_handle_Cdv.setData(1e4 * np.array(self.Cdv))
        self.parent.drag_graph.pg_plot_handle_Cdw.setData(1e4 * np.array(self.Cdw))


class CpPlotCallbackXFOIL(OptCallback):
    def __init__(self, parent, Cp):
        super().__init__(parent=parent)
        self.parent = parent
        self.Cp = Cp

    def exec_callback(self, theme: dict, grid: bool):
        if self.parent.Cp_graph is None:
            self.parent.Cp_graph = CpGraph(theme, grid=grid)
        tab_name = "Cp"
        if tab_name not in self.parent.dock_widget_names:
            self.parent.add_new_tab_widget(self.parent.Cp_graph.w, tab_name)
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
    def __init__(self, parent, Cp):
        super().__init__(parent=parent)
        self.parent = parent
        self.Cp = Cp
        # self.x_max = self.parent.mea.calculate_max_x_extent()
        self.x_max = 1.0

    def exec_callback(self, theme: dict, grid: bool):
        if self.parent.Cp_graph is None:
            self.parent.Cp_graph = CpGraph(theme, grid=grid)
        tab_name = "Cp"
        if tab_name not in self.parent.dock_widget_names:
            self.parent.add_new_tab_widget(self.parent.Cp_graph.w, tab_name)
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
