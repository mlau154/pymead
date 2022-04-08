import numpy as np
from pyairpar.core.anchor_point import AnchorPoint
from pyairpar.core.free_point import FreePoint
from pyairpar.symmetric.symmetric_base_airfoil_params import SymmetricBaseAirfoilParams
from pyairpar.core.airfoil import Airfoil
from pyairpar.core.airfoil import bezier
import typing
from copy import deepcopy


class SymmetricAirfoil(Airfoil):

    def __init__(self,
                 number_coordinates: int = 100,
                 base_airfoil_params: SymmetricBaseAirfoilParams = SymmetricBaseAirfoilParams(),
                 anchor_point_tuple: typing.Tuple[AnchorPoint, ...] = (),
                 free_point_tuple: typing.Tuple[FreePoint, ...] = ()
                 ):

        anchor_point_tuple = self.mirror_anchor_points(anchor_point_tuple)
        free_point_tuple = self.mirror_free_points(free_point_tuple, anchor_point_tuple)
        # for anchor_point in anchor_point_tuple:
        #     if anchor_point.previous_anchor_point == 'le':
        #         raise Warning('The definition of a SymmetricAirfoil requires that no anchor points can be defined '
        #                       'after the leading edge. Leaving \'le\' as the previous_anchor_point could produce '
        #                       'undesired results.')
        #
        # for free_point in free_point_tuple:
        #     if free_point.previous_anchor_point == 'le':
        #         raise Warning('The definition of a SymmetricAirfoil requires that no free points can be defined '
        #                       'after the leading edge. Leaving \'le\' as the previous_anchor_point could produce '
        #                       'undesired results.')

        super().__init__(number_coordinates, base_airfoil_params, anchor_point_tuple, free_point_tuple)

    @staticmethod
    def mirror_anchor_points(anchor_point_tuple: typing.Tuple[AnchorPoint, ...]):
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

    # def mirror_control_points(self):
    #     to_mirror = self.control_points[::-1]
    #     to_mirror = to_mirror[5:, :]
    #     mirror = to_mirror @ np.array([[1, 0], [0, -1]])
    #     self.control_points = np.vstack((self.control_points[:-4, :], mirror))
    #     self.needs_update = True
    #
    # def update_anchor_point_array(self):
    #     mirrored_point_array = np.array([])
    #     for key in self.anchor_point_order:
    #         xy = self.anchor_points[key]
    #         if key == 'te_1':
    #             self.anchor_point_array = xy
    #         elif key == 'te_2':
    #             pass
    #         elif key == 'le':
    #             self.anchor_point_array = np.row_stack((self.anchor_point_array, xy))
    #         else:
    #             self.anchor_point_array = np.row_stack((self.anchor_point_array, xy))
    #             xy_minus = xy.reshape(1, 2) @ np.array([[1, 0], [0, -1]])
    #             if len(mirrored_point_array) == 0:
    #                 mirrored_point_array = xy_minus
    #             else:
    #                 mirrored_point_array = np.vstack((xy_minus, mirrored_point_array))
    #     if len(mirrored_point_array) != 0:
    #         self.anchor_point_array = np.row_stack((self.anchor_point_array, mirrored_point_array))
    #     self.anchor_point_array = np.row_stack((self.anchor_point_array, self.anchor_points['te_2']))
    #
    # def generate_airfoil_coordinates(self):
    #     if self.C:
    #         self.C = []
    #     P_start_idx = 0
    #     for idx in range(len(self.anchor_point_order) - 2):
    #         P_length = self.N[self.anchor_point_order[idx]] + 1
    #         P_end_idx = P_start_idx + P_length
    #         P = self.control_points[P_start_idx:P_end_idx, :]
    #         C = bezier(P, self.nt)
    #         self.C.append(C)
    #         P_start_idx = P_end_idx - 1
    #     for idx in range(len(self.anchor_point_order) - 2 - 1, -1, -1):
    #         P_length = self.N[self.anchor_point_order[idx]] + 1
    #         P_end_idx = P_start_idx + P_length
    #         P = self.control_points[P_start_idx:P_end_idx, :]
    #         C = bezier(P, self.nt)
    #         self.C.append(C)
    #         P_start_idx = P_end_idx - 1
    #     coords = np.array([])
    #     curvature = np.array([])
    #     for idx in range(len(self.C)):
    #         if idx == 0:
    #             coords = np.column_stack((self.C[idx]['x'], self.C[idx]['y']))
    #             curvature = np.column_stack((self.C[idx]['x'], self.C[idx]['k']))
    #         else:
    #             coords = np.row_stack((coords, np.column_stack((self.C[idx]['x'][1:], self.C[idx]['y'][1:]))))
    #             curvature = np.row_stack((curvature, np.column_stack((self.C[idx]['x'][1:], self.C[idx]['k'][1:]))))
    #     self.coords = coords
    #     self.curvature = curvature
    #     return self.coords, self.C
    #
    # def update(self):
    #     self.add_anchor_points()
    #     self.add_free_points()
    #     self.extract_parameters()
    #     self.order_control_points()
    #     self.mirror_control_points()
    #     # self.add_mirrored_anchor_points()
    #     self.update_anchor_point_array()
    #     self.rotate(-self.alf.value)
    #     self.generate_airfoil_coordinates()
    #     self.needs_update = False
