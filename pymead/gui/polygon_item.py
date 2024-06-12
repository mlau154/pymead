import pyqtgraph as pg
import shapely.geometry
from PyQt6.QtCore import QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPicture, QPainter, QPolygonF, QPainterPath
from pyqtgraph.GraphicsScene.mouseEvents import HoverEvent


# Create a subclass of GraphicsObject.
# The only required methods are paint() and boundingRect()
# (see QGraphicsItem documentation)
class PolygonItem(pg.GraphicsObject):
    sigPolyEnter = pyqtSignal(object, float, float)
    sigPolyExit = pyqtSignal(object)

    def __init__(self, data, airfoil, gui_obj, pen=None, brush=None):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.pen = pen
        self.brush = brush
        if self.pen is None:
            self.pen = pg.mkPen(255, 255, 255, 0)
            self.brush = pg.mkBrush(66, 233, 245, 50)
        self.picture = None
        self.polygon = QPolygonF()
        self.gui_obj = gui_obj
        self.airfoil = airfoil
        self.generatePicture()

    def update_polygon(self):
        """
        Updates the polygon based on a new set of vertices (contained in ``self.data``)
        """
        if len(self.polygon) != 0:
            self.polygon.clear()
        for row in self.data:
            self.polygon.append(QPointF(row[0], row[1]))

    def generatePicture(self):
        """
        Pre-computes the QPicture for speed.
        """
        self.picture = QPicture()
        p = QPainter(self.picture)
        p.setPen(self.pen)
        p.setBrush(self.brush)
        self.update_polygon()
        p.drawPolygon(self.polygon)
        p.end()
        if self.gui_obj.permanent_widget.inviscid_cl_combo.currentText() == self.airfoil.name():
            self.gui_obj.single_airfoil_inviscid_analysis(plot_cp=False)

    def shape(self):
        """
        Defines the shape of the polygon as a ``QPainterPath`` so that the hover detection is more accurate. Note that
        by default, Qt simply uses the shape returned by ``boundingRect()`` if this method is not overridden.

        Returns
        =======
        QPainterPath
            Painter path for the polygon
        """
        path = QPainterPath()
        path.addPolygon(self.polygon)
        path.closeSubpath()
        return path

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        # boundingRect _must_ indicate the entire area that will be drawn on
        # or else we will get artifacts and possibly crashing.
        # (in this case, QPicture does all the work of computing the bounding rect for us)
        return QRectF(self.picture.boundingRect())

    def getCentroid(self):
        """
        Gets the centroid of the polygon using shapely

        Returns
        =======
        typing.List[float]
            List of [x,y] representing the centroid
        """
        shapely_poly = shapely.geometry.polygon.Polygon(shell=self.data)
        centroid = shapely_poly.centroid.xy
        return [centroid[0][0], centroid[1][0]]

    def hoverEvent(self, ev: HoverEvent):
        """
        Emits custom signals sigPolyEnter and sigPolyExit on airfoil polygon enter and exit

        Parameters
        ==========
        ev: pg.GraphicsScene.mouseEvents.HoverEvent
            Event emitted by pyqtgraph whenever the mouse hovers over the polygon defined by ``shape()``
        """
        centroid = self.getCentroid()
        if ev.isExit():
            self.sigPolyExit.emit(self.airfoil)
        elif ev.isEnter():
            self.sigPolyEnter.emit(self.airfoil, centroid[0], centroid[1])
