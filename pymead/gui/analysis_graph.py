import pyqtgraph as pg


class AnalysisGraph:
    def __init__(self, pen=None, size: tuple = (1000, 300), background_color: str = 'w'):
        pg.setConfigOptions(antialias=True)

        if pen is None:
            pen = pg.mkPen(color='cornflowerblue', width=2)

        self.w = pg.GraphicsLayoutWidget(show=True, size=size)
        # self.w.setWindowTitle('Airfoil')
        self.w.setBackground(background_color)

        self.v = self.w.addPlot(pen=pen)
        self.v.invertY(True)
        self.legend = self.v.addLegend(offset=(300, 20))
        # self.v.setAspectLocked()
