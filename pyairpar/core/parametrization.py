import numpy as np
from copy import deepcopy
from pyairpar.utils.transformations import rotate, translate
from pyairpar.core.param_setup import ParamSetup


class AirfoilParametrization:

    def __init__(self, param_setup: ParamSetup, _generate_airfoils: callable, args=None):
        """
        ### Description:

        `AirfoilParametrization` is `pyairpar`'s way of linking the shape, position, or angle of one or more `Airfoil`s
        to an external set of parameters. For example, an `Airfoil` could be placed a certain distance from another
        `Airfoil` by the value contained in a `Param`. See `pyairpar.examples` for example implementations.

        ### Args:

        `airfoil_tuple`: a `tuple` of `Airfoil`s. To specify a single airfoil, use `(Airfoil(),)`. At least one airfoil
        must be specified.

        ### Returns:

        An instance of the `AirfoilParametrization` class
        """
        self.param_setup = param_setup
        self._generate_airfoils = _generate_airfoils
        self.args = args

        self.airfoil_tuple = None

        self.n_mirrors = 0

    def generate_airfoils(self, *args, **kwargs):
        airfoil_tuple = self._generate_airfoils(self.param_setup.param_dict, *args, **kwargs)
        if isinstance(airfoil_tuple, tuple):
            self.airfoil_tuple = airfoil_tuple
        else:
            raise Exception(f'_generate_airfoils must return a tuple. Return type was {type(airfoil_tuple)}')

    def mirror(self, axis: np.ndarray or tuple, fixed_airfoil_idx: int, linked_airfoil_idx: int,
               fixed_anchor_point_range: tuple, starting_prev_anchor_point_str_linked: str):
        """
        ### Description:

        Sets a portion of a linked `pyairpar.core.airfoil.Airfoil` as the mirror across a specified axis of a portion
        of a fixed `pyairpar.core.airfoil.Airfoil`.

        ### Args:

        `axis`: **Option 1**: input a `np.ndarray` of `shape=(2, 2)` where the first row is comprised of the \\(x\\)-
        and \\(y\\)-coordinates of the "head" (first point) of a line, and the second row is comprised of the \\(x\\)-
        and  \\(y\\)-coordinates of the "tail" (second point) of a line. **Option 2**: input a `tuple` where the first
        item is a `float` value representing the angle of the line in radians, and the second line is a `np.ndarray` of
        `shape=(2)` representing the \\(x\\)- and \\(y\\)-coordinates of a point that lies along the line representing
        the axis. **Option 3** input a `tuple` where the first item is a `float` value representing the slope of the
        line, and the second item is a `float` value representing the y-intercept of the line.

        `fixed_airfoil_idx`: index, within the `airfoil_tuple`, of the airfoil whose portion is to be mirrored

        `linked_airfoil_idx`: index, within the `airfoil_tuple`, of the airfoil to which control points are to be
        added by a mirroring operation

        `fixed_anchor_point_range`: a 2-item `tuple` representing the range of `pyairpar.core.anchor_point.AnchorPoint`s
        to be mirrored. The first item should be a `str` representing the name of the first
        `pyairpar.core.anchor_point.AnchorPoint` in the fixed airfoil, and the second item should be a `str`
        representing the name of the final `pyairpar.core.anchor_point.AnchorPoint` in the fixed airfoil
        (counter-clockwise ordering)

        `starting_prev_anchor_point_str_linked`: The name (`str`) of the anchor point where the mirror should begin
        on the linked airfoil.
        """

        self.n_mirrors += 1

        # Calculate the slope (m) and y-intercept (b) of the axis line from the input axis
        if isinstance(axis, np.ndarray):
            # Option 1        y - y_1 = m * (x - x_1) => b - y_1 = -m * x_1 => b = -m * x_1 + y_1
            x1, x2, y1, y2 = axis[0, 0], axis[0, 1], axis[1, 0], axis[1, 1]
            m = (y2 - y1) / (x2 - x1)
            b = -m * x1 + y1
        elif isinstance(axis, tuple):
            if isinstance(axis[1], np.ndarray):
                # Option 2
                theta = axis[0]
                x1, y1 = axis[1][0], axis[1][1]
                m = np.tan(theta)
                b = (-m * x1 + y1)
            elif isinstance(axis[1], float):
                # Option 3
                m, b = axis[0], axis[1]
            else:
                raise ValueError('input \'axis\' was a tuple where the second element was of an invalid length. The'
                                 'length of the second element should be either 1 or 2.')
        else:
            raise Exception('Invalid reflection axis input')

        # Calculate the reflection matrix
        reflect_mat = 1 / (1 + m ** 2) * np.array([[1 - m ** 2, 2 * m, -2 * m * b],
                                                   [2 * m, m ** 2 - 1, 2 * b],
                                                   [0, 0, 1 + m ** 2]])
        # Reflect the relevant anchor points and free points about the input axis
        fixed_airfoil = self.airfoil_tuple[fixed_airfoil_idx]
        linked_airfoil = self.airfoil_tuple[linked_airfoil_idx]
        linked_airfoil_anchor_points = []
        start_idx = fixed_airfoil.anchor_point_order.index(fixed_anchor_point_range[0])
        end_idx = fixed_airfoil.anchor_point_order.index(fixed_anchor_point_range[1])
        previous_anchor_point_strings = [starting_prev_anchor_point_str_linked]
        for anchor_point_idx in range(end_idx + 1, start_idx + 1, -1):
            previous_anchor_point_strings.append(fixed_airfoil.anchor_point_order[anchor_point_idx - 1])
        str_count = 0
        for anchor_point_idx in range(end_idx, start_idx - 1, -1):  # start_idx, end_idx + 1
            fixed_anchor_point_str = fixed_airfoil.anchor_point_order[anchor_point_idx]
            # If not a trailing edge or leading edge anchor point (treat these separately):
            if fixed_airfoil.anchor_point_order[anchor_point_idx] not in ['te_1', 'le', 'te_2']:
                linked_airfoil_anchor_point = \
                    deepcopy(next((anchor_point for anchor_point in fixed_airfoil.anchor_point_tuple
                                   if anchor_point.name == fixed_anchor_point_str), None))

                x = fixed_airfoil.anchor_points[fixed_anchor_point_str][0]
                y = fixed_airfoil.anchor_points[fixed_anchor_point_str][1]
                new_x, new_y = rotate(x, y, -fixed_airfoil.alf.value)
                new_x, new_y = translate(new_x, new_y, fixed_airfoil.dx.value, fixed_airfoil.dy.value)

                xy = (reflect_mat @ np.array([[new_x, new_y, 1]]).T).T[0, :2]
                linked_airfoil_anchor_point.name = fixed_anchor_point_str + '_mirror_' + str(self.n_mirrors)
                previous_string = previous_anchor_point_strings[str_count]
                if previous_string in ['le', 'te_1', 'te_2']:
                    linked_airfoil_anchor_point.previous_anchor_point = previous_string
                else:
                    linked_airfoil_anchor_point.previous_anchor_point = previous_string + \
                                                                        '_mirror_' + str(self.n_mirrors)

                x, y = translate(xy[0], xy[1], -linked_airfoil.dx.value, -linked_airfoil.dy.value)
                x, y = rotate(x, y, linked_airfoil.alf.value)

                linked_airfoil_anchor_point.x.value = x
                linked_airfoil_anchor_point.y.value = y
                linked_airfoil_anchor_point.xy = np.array([x, y])
                linked_airfoil_anchor_point.phi.value -= theta

                linked_airfoil_anchor_point.set_all_as_linked()
                linked_airfoil_anchor_points.append(linked_airfoil_anchor_point)
            elif fixed_airfoil.anchor_point_order[anchor_point_idx] == 'le':
                pass
            elif fixed_airfoil.anchor_point_order[anchor_point_idx] == 'te_1':
                pass
            elif fixed_airfoil.anchor_point_order[anchor_point_idx] == 'te_2':
                pass
            str_count += 1
        linked_airfoil.anchor_point_tuple = list(linked_airfoil.anchor_point_tuple)
        linked_airfoil.anchor_point_tuple.extend(linked_airfoil_anchor_points)
        linked_airfoil.anchor_point_tuple = tuple(linked_airfoil.anchor_point_tuple)

        # Add free points - exclude the free points that have a previous_anchor_string equal to the end of the
        # fixed_anchor_point_range
        free_points_to_add = deepcopy([free_point for free_point in fixed_airfoil.free_point_tuple
                                       if free_point.previous_anchor_point + '_mirror_' + str(self.n_mirrors) in
                                       [anchor_point.name for anchor_point in linked_airfoil_anchor_points
                                        if anchor_point.name != fixed_anchor_point_range[1] +
                                        '_mirror_' + str(self.n_mirrors)]])
        fixed_anchor_point_str_list = fixed_airfoil.anchor_point_order[
                                      fixed_airfoil.anchor_point_order.index(fixed_anchor_point_range[0]):
                                      fixed_airfoil.anchor_point_order.index(fixed_anchor_point_range[1]) + 1]
        linked_anchor_point_str_list = [string + '_mirror_' + str(self.n_mirrors)
                                        if string not in ['le', 'te_1', 'te_2']
                                        else string for string in fixed_anchor_point_str_list[::-1]]
        for idx, free_point in reversed(list(enumerate(free_points_to_add))):
            free_point.previous_anchor_point = linked_anchor_point_str_list[fixed_anchor_point_str_list.index(free_point.previous_anchor_point)]
            free_point.x.value, free_point.y.value = rotate(
                free_point.x.value, free_point.y.value, -fixed_airfoil.alf.value)
            free_point.x.value += fixed_airfoil.dx.value
            free_point.y.value += fixed_airfoil.dy.value
            xy = (reflect_mat @ np.array([[free_point.x.value, free_point.y.value, 1]]).T).T[0, :2]
            free_point.x.value = xy[0]
            free_point.y.value = xy[1]
            free_point.x.value -= linked_airfoil.dx.value
            free_point.y.value -= linked_airfoil.dy.value
            free_point.x.value, free_point.y.value = rotate(
                free_point.x.value, free_point.y.value, linked_airfoil.alf.value)
            free_point.xy = np.array([free_point.x.value, free_point.y.value])
            free_point.set_all_as_linked()
        linked_airfoil.free_point_tuple = list(linked_airfoil.free_point_tuple)
        linked_airfoil.free_point_tuple.extend(free_points_to_add[::-1])
        linked_airfoil.free_point_tuple = tuple(linked_airfoil.free_point_tuple)
        linked_airfoil.update()

    def override_parameters(self, parameter_info_values: list, normalized: bool = False, *args, **kwargs):
        self.param_setup.override_parameters(parameter_info_values, normalized)
        self.generate_airfoils(*args, **kwargs)

    def clone(self):
        """
        ### Description:

        Clones an `pyairpar.core.airfoil.Airfoil`
        """
        pass
