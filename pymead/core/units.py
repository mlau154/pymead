import numpy as np


class Units:
    def __init__(self):
        self._BASE_LENGTH_UNIT = "m"
        self._BASE_AREA_UNIT = "m2"
        self._BASE_ANGLE_UNIT = "rad"

        self._LENGTH_UNIT = self._BASE_LENGTH_UNIT
        self._AREA_UNIT = self._BASE_AREA_UNIT
        self._ANGLE_UNIT = self._BASE_ANGLE_UNIT

        self._LENGTH_CONVERSIONS = {"in": 39.37007874015748, "mm": 1000., "cm": 100.}
        self._AREA_CONVERSIONS = {"in2": 1550.0031000062002, "mm2": 1.0e6, "cm2": 1.0e4}
        self._ANGLE_CONVERSIONS = {"deg": 57.29577951308232}

    def current_length_unit(self):
        return self._LENGTH_UNIT

    def set_current_length_unit(self, unit: str):
        self._LENGTH_UNIT = unit

    def convert_length_to_base(self, value: float, unit: str):
        if unit == self._BASE_LENGTH_UNIT:
            return value
        return value / self._LENGTH_CONVERSIONS[unit]

    def convert_length_from_base(self, value: float, unit: str):
        if unit == self._BASE_LENGTH_UNIT:
            return value
        return value * self._LENGTH_CONVERSIONS[unit]

    def current_area_unit(self):
        return self._AREA_UNIT

    def set_current_area_unit(self, unit: str):
        self._AREA_UNIT = unit

    def convert_area_to_base(self, value: float, unit: str):
        if unit == self._BASE_AREA_UNIT:
            return value
        return value / self._AREA_CONVERSIONS[unit]

    def convert_area_from_base(self, value: float, unit: str):
        if unit == self._BASE_AREA_UNIT:
            return value
        return value * self._AREA_CONVERSIONS[unit]

    def current_angle_unit(self):
        return self._ANGLE_UNIT

    def set_current_angle_unit(self, unit: str):
        self._ANGLE_UNIT = unit

    def convert_angle_to_base(self, value: float, unit: str):
        if unit == self._BASE_ANGLE_UNIT:
            return value
        return value / self._ANGLE_CONVERSIONS[unit]

    def convert_angle_from_base(self, value: float, unit: str):
        if unit == self._BASE_ANGLE_UNIT:
            return value
        return value * self._ANGLE_CONVERSIONS[unit]

    def pi(self):
        return self.convert_angle_from_base(np.pi, self.current_angle_unit())

    def sin(self, angle: float):
        return np.sin(self.convert_angle_to_base(angle, self.current_angle_unit()))

    def cos(self, angle: float):
        return np.cos(self.convert_angle_to_base(angle, self.current_angle_unit()))
