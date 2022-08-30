from PyQt5.QtWidgets import QToolBar, QToolButton
from PyQt5.QtGui import QIcon
from matplotlib.pyplot import imread
from PIL import Image

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent_dir = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent_dir)

from utils.read_write_json import read_json


class MainIconToolbar(QToolBar):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
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

    def on_grid_button_pressed(self, checked):
        if checked:
            self.parent.mplcanvas1.axes.grid(**self.grid_kwargs)
        else:
            self.parent.mplcanvas1.axes.grid(visible=False)
        self.parent.mplcanvas1.draw()

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

