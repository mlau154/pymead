from copy import deepcopy

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QDoubleSpinBox, QSpinBox, QLabel
from PyQt6.QtWidgets import QGridLayout

from pymead.core.geometry_collection import GeometryCollection
from pymead.optimization.sampling import ConstrictedRandomSampling


class SamplingVisualizationWidget(QWidget):
    def __init__(self, parent, geo_col_dict: dict, initial_sampling_width: float, initial_n_samples: int,
                 background_color: str):
        super().__init__(parent=parent)

        self.geo_col_dict = deepcopy(geo_col_dict)
        self.geo_col_dict["airfoil_graphs_active"] = False
        self.geo_col = GeometryCollection.set_from_dict_rep(self.geo_col_dict)
        norm_param_list = self.geo_col.extract_design_variable_values(bounds_normalized=True)
        self.norm_param_list = deepcopy(norm_param_list)

        self.w = pg.GraphicsLayoutWidget(show=True, size=(800, 300))
        # self.w.setWindowTitle('Airfoil')
        self.w.setBackground(background_color)

        self.v = self.w.addPlot()
        self.v.setLabel(axis="bottom", text="x")
        self.v.setLabel(axis="left", text="y")
        self.v.setAspectLocked(True)

        layout = QGridLayout(self)
        layout.addWidget(self.w, 0, 0, 1, 4)

        self.spinbox_ws = QDoubleSpinBox(self)
        self.spinbox_ns = QSpinBox(self)

        self.spinbox_ws.setMinimum(0.0)
        self.spinbox_ws.setMaximum(1.0)
        self.spinbox_ws.setDecimals(4)
        self.spinbox_ws.setSingleStep(0.01)
        self.spinbox_ws.setValue(initial_sampling_width)
        self.spinbox_ws.setKeyboardTracking(False)

        self.spinbox_ns.setMaximum(1000)
        self.spinbox_ns.setValue(initial_n_samples)
        self.spinbox_ns.setKeyboardTracking(False)

        self.spinbox_ws.valueChanged.connect(self.spinbox_ws_value_changed)
        self.spinbox_ns.valueChanged.connect(self.spinbox_ns_value_changed)

        self.slider_ws_label = QLabel(f"Sampling Width:", self)
        self.slider_ns_label = QLabel(f"Number of Samples:", self)

        layout.addWidget(self.slider_ws_label, 1, 0, 1, 1)
        layout.addWidget(self.spinbox_ws, 1, 1, 1, 1)
        layout.addWidget(self.slider_ns_label, 1, 2, 1, 1)
        layout.addWidget(self.spinbox_ns, 1, 3, 1, 1)

        self.update_plots(sampling_width=initial_sampling_width, n_samples=initial_n_samples)

        self.current_ws = initial_sampling_width
        self.current_ns = initial_n_samples

        self.setLayout(layout)
        self.setMinimumSize(800, 300)

    def spinbox_ws_value_changed(self, new_ws):
        self.current_ws = new_ws
        self.update_plots(sampling_width=self.current_ws, n_samples=self.current_ns)

    def spinbox_ns_value_changed(self, new_ns):
        self.current_ns = new_ns
        self.update_plots(sampling_width=self.current_ws, n_samples=self.current_ns)

    def update_plots(self, sampling_width: float, n_samples: int):
        self.v.clear()
        sampling = ConstrictedRandomSampling(n_samples=n_samples, norm_param_list=deepcopy(self.norm_param_list),
                                             max_sampling_width=sampling_width)
        X_list = sampling.sample()
        for idx, individual in enumerate(X_list):
            self.geo_col.assign_design_variable_values(individual, bounds_normalized=True)
            for a in self.geo_col.container()["airfoils"].values():
                coords = a.get_coords_selig_format()
                plot_handle = self.v.plot(pen=pg.mkPen(color="cornflowerblue"))
                plot_handle.setData(coords[:, 0], coords[:, 1])
