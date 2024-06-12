import pandas as pd
from PyQt6.QtWidgets import QTextEdit

from pymead.core.geometry_collection import GeometryCollection
from pymead.gui.dialogs import PymeadDialog


class AirfoilStatistics:
    def __init__(self, geo_col: GeometryCollection):
        self.geo_col = geo_col
        self.df = None
        self.data = None
        self.col_spaces = None
        self.generate_data()

    def generate_data(self):
        cols = ["Max. t/c", "x/c @ Max. t/c", "Area", "Self-Intersecting"]
        index = []
        self.data = {k: [] for k in cols}
        for airfoil_name, airfoil in self.geo_col.container()["airfoils"].items():
            thickness_data = airfoil.compute_thickness()
            self.data[cols[0]].append(thickness_data["t/c_max"])
            self.data[cols[1]].append(thickness_data["t/c_max_x/c_loc"])
            self.data[cols[2]].append(airfoil.compute_area())
            self.data[cols[3]].append(int(airfoil.check_self_intersection()))
            index.append(airfoil_name)
        self.df = pd.DataFrame(data=self.data, index=index)

    def convert_to_html(self, **kwargs):
        """Convenience function for Pandas DataFrame conversion to HTML"""
        return self.df.to_html(**kwargs)

    def convert_to_latex(self, **kwargs):
        """Convenience function for Pandas DataFrame conversion to LaTeX"""
        return self.df.to_latex(**kwargs)

    def generate_text_edit_widget(self, parent, **html_kwargs):
        text_edit = QTextEdit(parent)
        html = self.convert_to_html(**html_kwargs)
        text_edit.setHtml(html)
        text_edit.setMinimumWidth(500)
        text_edit.setReadOnly(True)
        return text_edit


class AirfoilStatisticsDialog(PymeadDialog):
    def __init__(self, parent, airfoil_stats: AirfoilStatistics, theme: dict):
        self.stats_widget = airfoil_stats.generate_text_edit_widget(parent=parent, float_format="{:.8f}".format)
        super().__init__(parent=parent, window_title="Airfoil Statistics", widget=self.stats_widget,
                         theme=theme)
