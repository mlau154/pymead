from PyQt6.QtWidgets import QGraphicsView


class CustomGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(self.DragMode.ScrollHandDrag)

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() > 0:
            scale = 1.25
        else:
            scale = .8
        self.scale(scale, scale)
