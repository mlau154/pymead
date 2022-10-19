import pyqtgraph as pg
from PyQt5.QtGui import QPicture, QPainter, QPolygonF
from PyQt5.QtCore import QPointF, QRectF


# Create a subclass of GraphicsObject.
# The only required methods are paint() and boundingRect()
# (see QGraphicsItem documentation)
class PolygonItem(pg.GraphicsObject):
    def __init__(self, data, pen=None, brush=None):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.pen = pen
        self.brush = brush
        if self.pen is None:
            self.pen = pg.mkPen(255, 255, 255, 0)
            self.brush = pg.mkBrush(66, 233, 245, 50)
        self.picture = None
        self.polygon = QPolygonF()
        self.generatePicture()

    def update_polygon(self):
        if len(self.polygon) != 0:
            self.polygon.clear()
        for row in self.data:
            self.polygon.append(QPointF(row[0], row[1]))

    def generatePicture(self):
        # pre-computing a QPicture object allows paint() to run much more quickly,
        # rather than re-drawing the shapes every time.
        self.picture = QPicture()
        p = QPainter(self.picture)
        p.setPen(self.pen)
        p.setBrush(self.brush)
        self.update_polygon()
        p.drawPolygon(self.polygon)
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        # boundingRect _must_ indicate the entire area that will be drawn on
        # or else we will get artifacts and possibly crashing.
        # (in this case, QPicture does all the work of computing the bounding rect for us)
        return QRectF(self.picture.boundingRect())


if __name__ == '__main__':
    data_ = [[0, 0], [0.1, 0.1], [0.2, 0.3], [-1, 1], [0, 0]]
    item = PolygonItem(data_)
    plt = pg.plot()
    plt.addItem(item)
    plt.setWindowTitle('pyqtgraph example: customGraphicsItem')
    pg.exec()
