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
        self.v.showGrid(x=True, y=True)
        self.legend = self.v.addLegend(offset=(30, 30))
        self.pg_plot_handle_Cd = self.v.plot(pen=pg.mkPen(color='indianred'), name='Cd', symbol='s', symbolBrush=pg.mkBrush('indianred'))
        self.pg_plot_handle_Cdp = self.v.plot(pen=pg.mkPen(color='gold'), name='Cdp', symbol='o', symbolBrush=pg.mkBrush('gold'))
        self.pg_plot_handle_Cdf = self.v.plot(pen=pg.mkPen(color='limegreen'), name='Cdf', symbol='t', symbolBrush=pg.mkBrush('limegreen'))
        self.pg_plot_handle_Cdv = self.v.plot(pen=pg.mkPen(color='mediumturquoise'), name='Cdv', symbol='d', symbolBrush=pg.mkBrush('mediumturquoise'))
        self.pg_plot_handle_Cdw = self.v.plot(pen=pg.mkPen(color='violet'), name='Cdw', symbol='x', symbolBrush=pg.mkBrush('violet'))

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
        self.v.showGrid(x=True, y=True)

    def set_background(self, background_color: str):
        self.w.setBackground(background_color)
