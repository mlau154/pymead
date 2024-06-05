import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QGridLayout

from pymead.core import UNITS

from pymead.utils.misc import get_setting


class AnalysisGraph:
    def __init__(self, theme: dict, pen=None, size: tuple = (1000, 300), background_color: str = 'w'):
        pg.setConfigOptions(antialias=True)

        if pen is None:
            pen = pg.mkPen(color='cornflowerblue', width=2)

        self.w = pg.GraphicsLayoutWidget(show=True, size=size)

        self.v = self.w.addPlot(pen=pen)
        self.v.invertY(True)
        self.legend = self.v.addLegend(offset=(300, 20))
        self.set_formatting(theme=theme)

    def set_background(self, background_color: str):
        self.w.setBackground(background_color)

    def set_formatting(self, theme: dict):
        self.w.setBackground(theme["graph-background-color"])
        label_font = f"{get_setting('axis-label-point-size')}pt {get_setting('axis-label-font-family')}"
        self.v.setLabel(axis="bottom", text=f"x [{UNITS.current_length_unit()}]", font=label_font,
                           color=theme["main-color"])
        self.v.setLabel(axis="left", text=f"Pressure Coefficient", font=label_font, color=theme["main-color"])
        tick_font = QFont(get_setting("axis-tick-font-family"), get_setting("axis-tick-point-size"))
        self.v.getAxis("bottom").setTickFont(tick_font)
        self.v.getAxis("left").setTickFont(tick_font)
        self.v.getAxis("bottom").setTextPen(theme["main-color"])
        self.v.getAxis("left").setTextPen(theme["main-color"])

    def set_legend_label_format(self, theme: dict):
        for _, label in self.legend.items:
            label.item.setHtml(f"<span style='font-family: DejaVu Sans; color: "
                               f"{theme['main-color']}; size: 8pt'>{label.text}</span>")


class ResidualGraph:
    def __init__(self, theme: dict, pen=None, size: tuple = (1000, 300)):
        pg.setConfigOptions(antialias=True)

        if pen is None:
            pen = pg.mkPen(color='cornflowerblue', width=2)

        self.w = pg.GraphicsLayoutWidget(show=True, size=size)

        self.v = self.w.addPlot(pen=pen)
        self.v.setLogMode(x=False, y=True)
        self.legend = self.v.addLegend(offset=(-5, 5))
        target_pen = self.make_target_pen(theme)

        # Add a line at y=1.0e-5 representing the target value for the residuals
        self.target_line = self.v.addLine(y=-5, pen=target_pen)

        # Dummy item to get the target line to add to the legend
        self.target_line_dummy_plot = self.v.plot(pen=target_pen, name="Target")

        # Set up the residual plot lines
        self.plot_items = [
            self.v.plot(pen=pg.mkPen(color=theme["residual-color-1"]), name="Density"),  # rms(dR): DRRMS in STATE.INC
            self.v.plot(pen=pg.mkPen(color=theme["residual-color-2"]), name="Grid Node"),  # rms(dA): DNRMS in STATE.INC
            self.v.plot(pen=pg.mkPen(color=theme["residual-color-3"]), name="Viscosity")  # rms(dV): DVRMS in STATE.INC
        ]

        # Set the formatting for the graph based on the current theme
        self.set_formatting(theme=theme)

    @staticmethod
    def make_target_pen(theme: dict):
        target_pen = pg.mkPen(color=theme["residual-target-color"])
        target_pen.setStyle(Qt.DashLine)
        return target_pen

    def set_background(self, background_color: str):
        self.w.setBackground(background_color)

    def set_formatting(self, theme: dict):
        self.w.setBackground(theme["graph-background-color"])
        label_font = f"{get_setting('axis-label-point-size')}pt {get_setting('axis-label-font-family')}"
        self.v.setLabel(axis="bottom", text=f"Iteration", font=label_font,
                           color=theme["main-color"])
        self.v.setLabel(axis="left", text=f"Residual Value", font=label_font, color=theme["main-color"])
        tick_font = QFont(get_setting("axis-tick-font-family"), get_setting("axis-tick-point-size"))
        self.v.getAxis("bottom").setTickFont(tick_font)
        self.v.getAxis("left").setTickFont(tick_font)
        self.v.getAxis("bottom").setTextPen(theme["main-color"])
        self.v.getAxis("left").setTextPen(theme["main-color"])
        for idx, plot_item in enumerate(self.plot_items):
            plot_item.setPen(pg.mkPen(color=theme[f"residual-color-{idx+1}"]))
        target_pen = self.make_target_pen(theme)
        self.target_line.setPen(target_pen)
        self.target_line_dummy_plot.setPen(target_pen)

    def set_legend_label_format(self, theme: dict):
        for _, label in self.legend.items:
            label.item.setHtml(f"<span style='font-family: DejaVu Sans; color: "
                               f"{theme['main-color']}; size: 8pt'>{label.text}</span>")


class SinglePolarGraph:
    def __init__(self, theme: dict, graph_color_key: str, x_axis_label: str, y_axis_label: str,
                 pen=None, size: tuple = (350, 300)):
        pg.setConfigOptions(antialias=True)

        if pen is None:
            pen = pg.mkPen(color='cornflowerblue', width=2)

        self.w = pg.GraphicsLayoutWidget(show=True, size=size)

        self.v = self.w.addPlot(pen=pen)

        # Set up the line
        self.plot_items = [self.v.plot(pen=pg.mkPen(color=theme[graph_color_key]))]

        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.graph_color_key = graph_color_key

        # Set the formatting for the graph based on the current theme
        self.set_formatting(theme=theme)

    def set_background(self, background_color: str):
        self.w.setBackground(background_color)

    def set_formatting(self, theme: dict):
        self.w.setBackground(theme["graph-background-color"])
        label_font = f"{get_setting('axis-label-point-size')}pt {get_setting('axis-label-font-family')}"
        self.v.setLabel(axis="bottom", text=self.x_axis_label, font=label_font,
                        color=theme["main-color"])
        self.v.setLabel(axis="left", text=self.y_axis_label, font=label_font, color=theme["main-color"])
        tick_font = QFont(get_setting("axis-tick-font-family"), get_setting("axis-tick-point-size"))
        self.v.getAxis("bottom").setTickFont(tick_font)
        self.v.getAxis("left").setTickFont(tick_font)
        self.v.getAxis("bottom").setTextPen(theme["main-color"])
        self.v.getAxis("left").setTextPen(theme["main-color"])
        for idx, plot_item in enumerate(self.plot_items):
            plot_item.setPen(pg.mkPen(color=theme[self.graph_color_key]))


class PolarGraphCollection(QWidget):
    def __init__(self, theme: dict):
        self.polar_graphs = [
            SinglePolarGraph(theme=theme, graph_color_key="polar-color-1",
                             x_axis_label="Angle of Attack (deg)", y_axis_label="Lift Coefficient"),
            SinglePolarGraph(theme=theme, graph_color_key="polar-color-2",
                             x_axis_label="Drag Coefficient", y_axis_label="Lift Coefficient"),
            SinglePolarGraph(theme=theme, graph_color_key="polar-color-3",
                             x_axis_label="Angle of Attack (deg)", y_axis_label="Lift-to-Drag Ratio"),
            SinglePolarGraph(theme=theme, graph_color_key="polar-color-4",
                             x_axis_label="Angle of Attack (deg)", y_axis_label="Pitching Moment Coefficient")
        ]
        super().__init__(parent=None)
        self.lay = QGridLayout()
        self.setLayout(self.lay)
        self.lay.addWidget(self.polar_graphs[0].w, 0, 0)
        self.lay.addWidget(self.polar_graphs[1].w, 0, 1)
        self.lay.addWidget(self.polar_graphs[2].w, 1, 0)
        self.lay.addWidget(self.polar_graphs[3].w, 1, 1)

    def set_background(self, background_color: str):
        for polar_graph in self.polar_graphs:
            polar_graph.set_background(background_color)

    def set_formatting(self, theme: dict):
        for polar_graph in self.polar_graphs:
            polar_graph.set_formatting(theme)

    def set_data(self, aero_data: dict):
        lift_to_drag_ratio = (np.array(aero_data["Cl"]) / np.array(aero_data["Cd"])).tolist()
        self.polar_graphs[0].plot_items[0].setData(x=aero_data["alf"], y=aero_data["Cl"])
        self.polar_graphs[1].plot_items[0].setData(x=aero_data["Cd"], y=aero_data["Cl"])
        self.polar_graphs[2].plot_items[0].setData(x=aero_data["alf"], y=lift_to_drag_ratio)
        self.polar_graphs[3].plot_items[0].setData(x=aero_data["alf"], y=aero_data["Cm"])

    def clear_data(self):
        for polar_graph in self.polar_graphs:
            polar_graph.plot_items[0].setData([], [])
