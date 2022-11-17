import numpy as np
from PyQt5.QtWidgets import QToolBar, QToolButton
from PyQt5.QtGui import QIcon
from pymead import DATA_DIR
import os
from pymead.core.airfoil import Airfoil
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.param import Param
from pymead import ICON_DIR
from functools import partial


class MainIconToolbar(QToolBar):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        self.new_airfoil_location = None
        self.icon_dir = ICON_DIR
        self.parent.addToolBar(self)
        self.grid_icon = QIcon(os.path.join(self.icon_dir, 'grid_icon.png'))
        self.grid_button = QToolButton(self)
        self.grid_button.setStatusTip("Activate grid")
        self.grid_button.setCheckable(True)
        self.grid_button.setIcon(self.grid_icon)
        self.grid_button.toggled.connect(self.on_grid_button_pressed)
        # self.grid_kwargs = read_json(os.path.join('gui_settings', 'grid_settings.json'))
        self.addWidget(self.grid_button)

        # self.add_image_icon = QIcon(os.path.join(self.icon_dir, 'add_image.png'))
        # self.add_image_button = QToolButton(self)
        # self.add_image_button.setStatusTip("Add background to image")
        # self.add_image_button.setCheckable(True)
        # self.add_image_button.setIcon(self.add_image_icon)
        # self.add_image_button.toggled.connect(self.add_image_button_toggled)
        # self.addWidget(self.add_image_button)

        self.change_background_color_icon = QIcon(os.path.join(self.icon_dir, 'color_palette.png'))
        self.change_background_color_button = QToolButton(self)
        self.change_background_color_button.setStatusTip("Change background color")
        self.change_background_color_button.setCheckable(True)
        self.change_background_color_button.setIcon(self.change_background_color_icon)
        self.change_background_color_button.clicked.connect(self.change_background_color_button_toggled)
        self.addWidget(self.change_background_color_button)

        self.add_airfoil_icon = QIcon(os.path.join(self.icon_dir, 'Add-icon3.png'))
        self.add_airfoil_button = QToolButton(self)
        self.add_airfoil_button.setStatusTip("Add airfoil (click on the graph to complete action)")
        self.add_airfoil_button.setCheckable(False)
        self.add_airfoil_button.setIcon(self.add_airfoil_icon)
        self.add_airfoil_button.clicked.connect(self.add_airfoil_button_toggled)
        self.addWidget(self.add_airfoil_button)

        self.te_thickness_icon = QIcon(os.path.join(self.icon_dir, 'thickness_icon.png'))
        self.te_thickness_button = QToolButton(self)
        self.te_thickness_button.setStatusTip("Toggle trailing edge thickness edit mode")
        self.te_thickness_button.setCheckable(True)
        self.te_thickness_button.setIcon(self.te_thickness_icon)
        self.te_thickness_button.toggled.connect(self.te_thickness_mode_toggled)
        self.addWidget(self.te_thickness_button)

    def on_grid_button_pressed(self, checked):
        # import pyqtgraph as pg
        if checked:
            self.parent.v.showGrid(x=True, y=True)
        else:
            self.parent.v.showGrid(x=False, y=False)
        # h = self.parent.copy_mea()
        # print(f"current mea is located at {hex(id(self.parent.mea))}")
        # print(f"dill-copied mea is located at {hex(id(h))}")
        # internal_geometry_xy = np.loadtxt(os.path.join(DATA_DIR, 'sec_6.txt'))
        # scale_factor = 0.612745
        # x_start = 0.13352022
        # internal_geometry = self.parent.v.plot(internal_geometry_xy[:, 0] * scale_factor + x_start,
        #                                        internal_geometry_xy[:, 1] * scale_factor,
        #                                        pen=pg.mkPen(color='orange', width=1))
        # self.parent.mea.extract_parameters()
        # parameter_list = np.loadtxt(os.path.join(DATA_DIR, 'parameter_list.dat'))
        # self.parent.mea.update_parameters(parameter_list)
        # Need to now update the parameter tree and airfoil graph to reflect these changes
        # fig_, axs_ = plt.subplots()
        # for a in self.parent.mea.airfoils.values():
        #     # a.update()
        #     a.plot_airfoil(axs=axs_)
        #     a.plot_control_points(axs=axs_, marker='o', color='gray')
        # axs_.set_aspect('equal')
        # plt.show()


    # def add_image_button_toggled(self, checked):
    #     if checked:
    #         image = os.path.join(self.icon_dir, 'airfoil_slat.png')
    #         fig = self.parent.mplcanvas1.figure
    #         ax = self.parent.mplcanvas1.axes
    #         xlimits = ax.get_xlim()
    #         ylimits = ax.get_ylim()
    #         extent = [xlimits[0], xlimits[1], ylimits[0], ylimits[1]]
    #         # print(f"extent = {self.parent.mplcanvas1.axes.get_xlim()}")
    #         with Image.open(image) as im:
    #             self.figure_to_remove = self.parent.mplcanvas1.axes.imshow(im, extent=extent)
    #     else:
    #         self.figure_to_remove.remove()
    #     self.parent.mplcanvas1.draw()

    def change_background_color_button_toggled(self, checked):
        if checked:
            self.parent.dark_mode = True
            self.parent.set_dark_mode()
        else:
            self.parent.dark_mode = False
            self.parent.set_light_mode()

        if self.parent.analysis_graph is not None:
            if checked:
                self.parent.analysis_graph.set_background('#2a2a2b')
            else:
                self.parent.analysis_graph.set_background('w')
        if checked:
            self.parent.param_tree_instance.set_dark_mode()
        else:
            self.parent.param_tree_instance.set_light_mode()
        # QTreeView::branch:closed {color: white;} QTreeView::branch:open {color: white;}''')
        # QTreeView::item {border: 1px solid black;}''')  # need to use image, not
        # color for open closed arrows
        # self.parent.design_tree_widget.updatePalette()
        self.parent.show()

    def add_airfoil_button_toggled(self):
        def scene_clicked(ev):
            self.new_airfoil_location = self.parent.mea.v.vb.mapSceneToView(ev.scenePos())
            airfoil = Airfoil(base_airfoil_params=BaseAirfoilParams(dx=Param(self.new_airfoil_location.x()),
                                                                    dy=Param(self.new_airfoil_location.y())))
            self.parent.mea.te_thickness_edit_mode = self.parent.te_thickness_edit_mode
            self.parent.mea.add_airfoil(airfoil, len(self.parent.mea.airfoils), self.parent.param_tree_instance)
            self.parent.param_tree_instance.p.child("Analysis").child("Inviscid Cl Calc").setLimits([a.tag for a in self.parent.mea.airfoils.values()])
            self.parent.param_tree_instance.params[-1].add_airfoil(airfoil, len(self.parent.mea.airfoils) - 1)
            self.parent.mea.v.scene().sigMouseClicked.disconnect()
            airfoil.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.parent.param_tree_instance.plot_changed, f"A{len(self.parent.mea.airfoils) - 1}"))

        self.parent.mea.v.scene().sigMouseClicked.connect(scene_clicked)

    def te_thickness_mode_toggled(self, checked):
        if checked:
            for airfoil in self.parent.mea.airfoils.values():
                airfoil.airfoil_graph.te_thickness_edit_mode = True
            self.parent.te_thickness_edit_mode = True
        else:
            for airfoil in self.parent.mea.airfoils.values():
                airfoil.airfoil_graph.te_thickness_edit_mode = False
            self.parent.te_thickness_edit_mode = False
