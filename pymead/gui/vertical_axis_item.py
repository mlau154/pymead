import math

import pyqtgraph as pg
from pyqtgraph import debug as debug


class AngledAxisItem(pg.AxisItem):
    def __init__(self, angle: float = 0.0, *args, **kwargs):
        """
        Custom implementation of ``pyqtgraph.AxisItem`` adapted from gnomezgrave's answer
        (https://github.com/pyqtgraph/pyqtgraph/issues/322#issuecomment-503541303) in ``pyqtgraph``
        issue #322 (https://github.com/pyqtgraph/pyqtgraph/issues/322).

        Parameters
        ----------
        angle: float
            Angle by which to rotate the text. Default: ``0.0``
        args:
            Arguments to pass to ``pyqtgraph.AxisItem.__init__``
        kwargs
            Keyword arguments to pass to ``pyqtgraph.AxisItem.__init__``
        """
        self.angle = angle
        self._height_updated = False
        super().__init__(*args, **kwargs)

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):

        profiler = debug.Profiler()

        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setRenderHint(p.RenderHint.TextAntialiasing, True)

        # draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)

        # draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)
        profiler('draw ticks')

        # Draw all text
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        p.setPen(self.textPen())
        # bounding = self.boundingRect().toAlignedRect()
        # p.setClipRect(bounding)

        max_width = 0

        for rect, flags, text in textSpecs:
            p.save()  # save the painter state

            p.translate(rect.center())  # move coordinate system to center of text rect
            p.rotate(self.angle)  # rotate text
            p.translate(-rect.center())  # revert coordinate system

            x_offset = math.ceil(math.fabs(math.sin(math.radians(self.angle)) * rect.width()))
            if self.angle < 0:
                x_offset = -x_offset
            p.translate(x_offset / 2, 0)  # Move the coordinate system (relatively) downwards

            p.drawText(rect, flags, text)
            p.restore()  # restore the painter state
            offset = math.fabs(x_offset)
            max_width = offset if max_width < offset else max_width

        profiler('draw text')
        #  Adjust the height
        if not self._height_updated:
            self.setHeight(self.height() + max_width)
            self._height_updated = True
