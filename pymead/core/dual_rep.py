from copy import deepcopy


class DualRep:
    """
    Dual-representation class. This class is sub-classed by any objects in ``pymead`` that require representation in
    both the API and GUI layers. Purely used to allow the deepcopy action to work on ``pymead`` objects that contain
    non-serializable attributes, such as ``Qt`` slots, signals, or graphics items. Note that the ``__init__`` method
    is not implemented, which means that this class can be used as a parent without requiring a call to
    ``super().__init__()`` in any of the child classes.
    """

    # These attributes are non-serializable because they contain references to PyQt5 graphics or signal/slot objects,
    # which are inherently non-serializable
    non_serializable_attributes = ["tree_item", "gui_obj", "geo_canvas", "geo_tree"]

    def __deepcopy__(self, memo):
        """
        Overwrite the ``deepcopy`` method to set any non-serializable attributes in the copied object to ``None``.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():

            # Eliminate any GUI references so the object is deep-copyable
            if k in self.non_serializable_attributes:
                v = None

            setattr(result, k, deepcopy(v, memo))
        return result
