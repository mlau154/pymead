import pyqtgraph as pg
from PyQt5.QtGui import QFont
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
