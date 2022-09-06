from PyQt5.QtWidgets import QToolBar, QToolButton
from PyQt5.QtGui import QIcon
from matplotlib.pyplot import imread
from PIL import Image

import sys
import os

from pyairpar.utils.read_write_json import read_json
from pyairpar.gui.airfoil_graph import AirfoilGraph
from pyairpar.core.airfoil import Airfoil
from pyairpar.core.base_airfoil_params import BaseAirfoilParams
from pyairpar.core.param import Param


class MainIconToolbar(QToolBar):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        self.new_airfoil_location = None
        self.icon_dir = os.path.join(os.path.dirname(os.getcwd()), 'icons')
        self.parent.addToolBar(self)
        self.grid_icon = QIcon(os.path.join(self.icon_dir, 'grid_icon.png'))
        self.grid_button = QToolButton(self)
        self.grid_button.setStatusTip("Activate grid")
        self.grid_button.setCheckable(True)
        self.grid_button.setIcon(self.grid_icon)
        self.grid_button.toggled.connect(self.on_grid_button_pressed)
        self.grid_kwargs = read_json(os.path.join('gui_settings', 'grid_settings.json'))
        self.addWidget(self.grid_button)

        self.add_image_icon = QIcon(os.path.join(self.icon_dir, 'add_image.png'))
        self.add_image_button = QToolButton(self)
        self.add_image_button.setStatusTip("Add background to image")
        self.add_image_button.setCheckable(True)
        self.add_image_button.setIcon(self.add_image_icon)
        self.add_image_button.toggled.connect(self.add_image_button_toggled)
        self.addWidget(self.add_image_button)

        self.change_background_color_icon = QIcon(os.path.join(self.icon_dir, 'color_palette.png'))
        self.change_background_color_button = QToolButton(self)
        self.change_background_color_button.setStatusTip("Change background color")
        self.change_background_color_button.setCheckable(False)
        self.change_background_color_button.setIcon(self.change_background_color_icon)
        self.change_background_color_button.clicked.connect(self.change_background_color_button_toggled)
        self.addWidget(self.change_background_color_button)

        self.add_airfoil_icon = QIcon(os.path.join(self.icon_dir, 'add_icon.png'))
        self.add_airfoil_button = QToolButton(self)
        self.add_airfoil_button.setStatusTip("Add airfoil")
        self.add_airfoil_button.setCheckable(False)
        self.add_airfoil_button.setIcon(self.add_airfoil_icon)
        self.add_airfoil_button.clicked.connect(self.add_airfoil_button_toggled)
        self.addWidget(self.add_airfoil_button)

    def on_grid_button_pressed(self, checked):
        if checked:
            self.parent.v.showGrid(x=True, y=True)
        else:
            self.parent.v.showGrid(x=False, y=False)

    def add_image_button_toggled(self, checked):
        if checked:
            image = os.path.join(self.icon_dir, 'airfoil_slat.png')
            fig = self.parent.mplcanvas1.figure
            ax = self.parent.mplcanvas1.axes
            xlimits = ax.get_xlim()
            ylimits = ax.get_ylim()
            extent = [xlimits[0], xlimits[1], ylimits[0], ylimits[1]]
            # print(f"extent = {self.parent.mplcanvas1.axes.get_xlim()}")
            with Image.open(image) as im:
                self.figure_to_remove = self.parent.mplcanvas1.axes.imshow(im, extent=extent)
        else:
            self.figure_to_remove.remove()
        self.parent.mplcanvas1.draw()

    def change_background_color_button_toggled(self):
        self.parent.setStyleSheet("background-color: black;")
        self.parent.w.setBackground('k')
        self.parent.design_tree.setStyleSheet('''QTreeWidget {color: white;} QTreeView::item:hover {background: green;}
        QTreeView::branch:closed {color: white;} QTreeView::branch:open {color: white;}''')  # need to use image, not
        # color for open closed arrows
        self.parent.show()

    def add_airfoil_button_toggled(self):
        def scene_clicked(ev):
            self.new_airfoil_location = self.parent.v.vb.mapSceneToView(ev.scenePos())
            airfoil = Airfoil(base_airfoil_params=BaseAirfoilParams(dx=Param(self.new_airfoil_location.x()),
                                                                    dy=Param(self.new_airfoil_location.y())))
            self.parent.airfoil_graphs.append(AirfoilGraph(airfoil=airfoil, w=self.parent.w, v=self.parent.v))
            self.parent.v.scene().sigMouseClicked.disconnect()

        self.parent.v.scene().sigMouseClicked.connect(scene_clicked)
