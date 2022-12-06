from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtCore import Qt


class CustomGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(self.ScrollHandDrag)

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() > 0:
            scale = 1.25
        else:
            scale = .8
        self.scale(scale, scale)
