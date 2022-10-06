from matplotlib.pyplot import get_cmap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


class MplColorHelper:
    """
    From https://stackoverflow.com/questions/26108436/how-can-i-get-the-matplotlib-rgb-color-given-the-colormap-name
    -boundrynorm-an
    """
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = get_cmap(cmap_name)
        self.norm = Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        rgba_tuple = self.scalarMap.to_rgba(val)
        rgba_list = list(rgba_tuple)
        rgba_256_int = tuple([int(elem * 255) for elem in rgba_list])
        return rgba_256_int
