# Draggable rectangle with blitting.
import numpy as np
import matplotlib.pyplot as plt
import typing


class DraggableLine:
    # Adapted from Matplotlib documentation:
    # https://matplotlib.org/stable/users/explain/event_handling.html

    lock = None  # only one can be animated at a time

    def __init__(self, line, detection_radius_pixels=7, button_release_callback: typing.Callable = None, **kwargs):
        self.line = line
        self.dx = None
        self.dy = None
        self.radius = detection_radius_pixels
        self.ind = None
        self.press = None
        self.background = None
        self.button_release_callback = button_release_callback
        self.kwargs = kwargs

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.line.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if (event.inaxes != self.line.axes
                or DraggableLine.lock is not None):
            return
        contains, attrd = self.line.contains(event)
        if not contains:
            return
        x, y = self.line.get_data()

        xy_pixels = self.line.axes.transData.transform(np.column_stack((x, y)))

        pixel_distance_array = np.sqrt((xy_pixels[:, 0] - event.x)**2 + (xy_pixels[:, 1] - event.y)**2)

        min_distance = np.min(pixel_distance_array)

        if min_distance > self.radius:
            return

        idx_of_point_to_drag = np.argmin(pixel_distance_array)
        self.ind = idx_of_point_to_drag

        self.press = (x[idx_of_point_to_drag], y[idx_of_point_to_drag]), (event.xdata, event.ydata)
        DraggableLine.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.line.figure.canvas
        axes = self.line.axes
        self.line.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.line.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.line)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if (event.inaxes != self.line.axes
                or DraggableLine.lock is not self):
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        xdata, ydata = self.line.get_data()
        xdata[self.ind] = x0 + dx
        ydata[self.ind] = y0 + dy
        self.dx = dx
        self.dy = dy
        self.line.set_xdata(xdata)
        self.line.set_ydata(ydata)

        canvas = self.line.figure.canvas
        axes = self.line.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.line)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        """Clear button press information."""
        if DraggableLine.lock is not self:
            return

        self.press = None
        DraggableLine.lock = None

        # turn off the rect animation property and reset the background
        self.line.set_animated(False)
        self.background = None

        x, y = self.line.get_data()

        if self.button_release_callback is not None:
            self.button_release_callback(self.ind, x, y, self.line, self.dx, self.dy,  **self.kwargs)

        # redraw the full figure
        self.line.figure.canvas.draw()
        print(self.line)

    def disconnect(self):
        """Disconnect all callbacks."""
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)


def main():
    fig, ax = plt.subplots()
    lines = ax.plot([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], color='gold', marker='o')
    drs = []
    for line in lines:
        dr = DraggableLine(line, 7)
        dr.connect()
        drs.append(dr)

    plt.show()


if __name__ == '__main__':
    main()
