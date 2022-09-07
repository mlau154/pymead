import numpy as np
from pymead.core.anchor_point import AnchorPoint
from pymead.core.free_point import FreePoint
from pymead.symmetric.symmetric_base_airfoil_params import SymmetricBaseAirfoilParams
from pymead.core.airfoil import Airfoil
import typing
from copy import deepcopy


class SymmetricAirfoil(Airfoil):

    def __init__(self,
                 number_coordinates: int = 100,
                 base_airfoil_params: SymmetricBaseAirfoilParams = SymmetricBaseAirfoilParams(),
                 anchor_point_tuple: typing.Tuple[AnchorPoint, ...] = (),
                 free_point_tuple: typing.Tuple[FreePoint, ...] = ()
                 ):
        """
        ### Description:

        A sub-class of `pymead.core.airfoil.Airfoil` that describes a symmetric airfoil using a reduced parameter set
        (`pymead.symmetric.symmetric_base_airfoil_params.SymmetricBaseAirfoilParams`)
        """
        anchor_point_tuple = self.mirror_anchor_points(anchor_point_tuple)
        free_point_tuple = self.mirror_free_points(free_point_tuple, anchor_point_tuple)

        super().__init__(number_coordinates, base_airfoil_params, anchor_point_tuple, free_point_tuple)

    @staticmethod
    def mirror_anchor_points(anchor_point_tuple: typing.Tuple[AnchorPoint, ...]):
        """
        ### Description:

        Mirrors all anchor points in the `anchor_point_tuple` across the chordline.

        ### Returns:

        The modified `anchor_point_tuple`
        """
        # The next 3 lines produce, for example, ['te_1', 'ap1', 'ap2', 'ap3', 'le']
        upper_anchor_point_string_list = [anchor_point.name for anchor_point in anchor_point_tuple]
        upper_anchor_point_string_list.insert(0, 'te_1')
        upper_anchor_point_string_list.append('le')
        # The next line produces, for example, ['le', 'ap3', 'ap2']
        previous_anchor_strings = upper_anchor_point_string_list[-1:1:-1]
        # The next produces, for example, ['ap3', 'ap2', 'ap1']
        names = upper_anchor_point_string_list[-2:0:-1]
        add_these_anchor_points = []
        for idx, anchor_point in enumerate(anchor_point_tuple[::-1]):
            mirrored_anchor_point = deepcopy(anchor_point)
            mirrored_anchor_point.previous_anchor_point = previous_anchor_strings[idx]
            if mirrored_anchor_point.previous_anchor_point != 'le':
                mirrored_anchor_point.previous_anchor_point = mirrored_anchor_point.previous_anchor_point + '_symm'
            mirrored_anchor_point.name = names[idx] + '_symm'
            mirrored_anchor_point.y.value = -mirrored_anchor_point.y.value  # Reflect about the chordline (x-axis)
            mirrored_anchor_point.xy = np.array([mirrored_anchor_point.x.value, mirrored_anchor_point.y.value])
            mirrored_anchor_point.set_all_as_linked()
            add_these_anchor_points.append(mirrored_anchor_point)
        anchor_point_tuple = list(anchor_point_tuple)
        anchor_point_tuple.extend(add_these_anchor_points)
        anchor_point_tuple = tuple(anchor_point_tuple)
        return anchor_point_tuple

    @staticmethod
    def mirror_free_points(free_point_tuple: typing.Tuple[FreePoint, ...],
                           anchor_point_tuple: typing.Tuple[AnchorPoint, ...]):
        """
        ### Description:

        Mirrors all free points in the `free_point_tuple` across the chordline.

        ### Returns:

        The modified `free_point_tuple`
        """
        # The next 3 lines produce, for example, ['te_1', 'ap1', 'ap2', 'ap3', 'le']
        upper_anchor_point_string_list = [
            anchor_point.name for anchor_point in anchor_point_tuple if '_symm' not in anchor_point.name]
        upper_anchor_point_string_list.insert(0, 'te_1')
        upper_anchor_point_string_list.append('le')
        # The next line produces, for example, ['te_1', 'ap1', 'ap2', 'ap3']
        previous_free_strings = upper_anchor_point_string_list[-2::-1]
        # The next produces, for example, ['ap3', 'ap2', 'ap1', 'le']
        previous_free_string_new = upper_anchor_point_string_list[:0:-1]
        upper_free_point_string_list = [free_point.previous_anchor_point for free_point in free_point_tuple]

        add_these_free_points = []
        for idx, free_point in enumerate(free_point_tuple[::-1]):
            mirrored_free_point = deepcopy(free_point)
            mirrored_free_point.previous_anchor_point = previous_free_string_new[
                previous_free_strings.index(mirrored_free_point.previous_anchor_point)]
            if mirrored_free_point.previous_anchor_point != 'le':
                mirrored_free_point.previous_anchor_point = mirrored_free_point.previous_anchor_point + '_symm'
            mirrored_free_point.y.value = -mirrored_free_point.y.value
            mirrored_free_point.xy = np.array([mirrored_free_point.x.value, mirrored_free_point.y.value])
            mirrored_free_point.set_all_as_linked()
            add_these_free_points.append(mirrored_free_point)
        free_point_tuple = list(free_point_tuple)
        free_point_tuple.extend(add_these_free_points)
        free_point_tuple = tuple(free_point_tuple)
        return free_point_tuple
