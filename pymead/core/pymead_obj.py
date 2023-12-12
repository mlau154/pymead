from copy import deepcopy
from abc import ABC, abstractmethod, abstractclassmethod


class DualRep:

    # These attributes are non-serializable because they contain references to PyQt5 graphics or signal/slot objects,
    # which are inherently non-serializable
    non_serializable_attributes = ["tree", "tree_item", "canvas", "canvas_item"]

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


class PymeadObj(ABC, DualRep):
    """
    Base class for all objects in pymead.
    """

    def __init__(self, sub_container: str):
        self.sub_container = sub_container
        self._name = None
        self.geo_col = None
        self.tree_item = None
        self.canvas_item = None

    def name(self):
        """
        Retrieves the parameter name

        Returns
        =======
        str
            The parameter name
        """
        return self._name

    def set_name(self, name: str):
        """
        Sets the object name.

        Parameters
        ==========
        name: str
            The object name
        """
        if self.geo_col is not None:
            # Rename the reference in the geometry collection if necessary
            sub_container = self.geo_col.container()[self.sub_container]

            if self.name() in sub_container and sub_container[self.name()] is self:
                sub_container[name] = sub_container[self.name()]
                sub_container.pop(self.name())

        self._name = name

    @abstractmethod
    def get_dict_rep(self):
        pass
