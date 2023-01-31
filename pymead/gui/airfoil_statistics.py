from pymead.core.mea import MEA
import pandas as pd
from PyQt5.QtWidgets import QTextEdit, QDialog, QVBoxLayout


class AirfoilStatistics:
    def __init__(self, mea: MEA):
        self.mea = mea
        self.df = None
        self.data = None
        self.col_spaces = None
        self.generate_data()

    def generate_data(self):
        cols = ['Max. t/c', 'x/c @ Max. t/c', 'Area', 'Self-Intersecting']
        index = []
        self.data = {k: [] for k in cols}
        for airfoil_name, airfoil in self.mea.airfoils.items():
            thickness_data = airfoil.compute_thickness(return_max_thickness_loc=True)
            self.data[cols[0]].append(thickness_data['t/c_max'])
            self.data[cols[1]].append(thickness_data['t/c_max_x/c_loc'])
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
        text_edit.setMinimumWidth(400)
        text_edit.setReadOnly(True)
        return text_edit


class AirfoilStatisticsDialog(QDialog):
    def __init__(self, parent, airfoil_stats: AirfoilStatistics):
        super().__init__(parent=parent)
        self.setWindowTitle("Airfoil Statistics")
        self.setFont(self.parent().font())

        layout = QVBoxLayout(self)
        self.stats_widget = airfoil_stats.generate_text_edit_widget(parent=parent, float_format='{:.8f}'.format)
        layout.addWidget(self.stats_widget)