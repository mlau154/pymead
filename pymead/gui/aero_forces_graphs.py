import pyqtgraph as pg


class DragGraph:
    def __init__(self, pen=None, size: tuple = (1000, 300), background_color: str = 'w'):
        pg.setConfigOptions(antialias=True)

        if pen is None:
            pen = pg.mkPen(color='cornflowerblue', width=2)

        self.w = pg.GraphicsLayoutWidget(show=True, size=size)
        # self.w.setWindowTitle('Airfoil')
        self.w.setBackground(background_color)
        self.v = self.w.addPlot(pen=pen)
        self.v.setLabel(axis='bottom', text='Generation')
        self.v.setLabel(axis='left', text='Drag (Counts)')
        self.pg_plot_handle_Cd = self.v.plot(pen=pg.mkPen(color='indianred'), name='Cd')
        self.pg_plot_handle_Cdp = self.v.plot(pen=pg.mkPen(color='gold'), name='Cdp')
        self.pg_plot_handle_Cdf = self.v.plot(pen=pg.mkPen(color='limegreen'), name='Cdf')
        self.legend = self.v.addLegend(offset=(300, 20))

    def set_background(self, background_color: str):
        self.w.setBackground(background_color)


class CpGraph:
    def __init__(self, pen=None, size: tuple = (1000, 300), background_color: str = 'w'):
        pg.setConfigOptions(antialias=True)

        if pen is None:
            pen = pg.mkPen(color='cornflowerblue', width=2)

        self.w = pg.GraphicsLayoutWidget(show=True, size=size)
        # self.w.setWindowTitle('Airfoil')
        self.w.setBackground(background_color)
        self.v = self.w.addPlot(pen=pen)
        self.v.setLabel(axis='bottom', text='x')
        self.v.setLabel(axis='left', text='Cp')
        self.v.invertY(True)
        self.legend = self.v.addLegend(offset=(300, 20))

    def set_background(self, background_color: str):
        self.w.setBackground(background_color)
