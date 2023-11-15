"""
``pymead.core`` contains the main modules that run `pymead`.
"""

_BASE_LENGTH_UNIT = "m"
_BASE_AREA_UNIT = "m2"
_BASE_ANGLE_UNIT = "rad"

_LENGTH_UNIT = _BASE_LENGTH_UNIT
_AREA_UNIT = _BASE_AREA_UNIT
_ANGLE_UNIT = _BASE_ANGLE_UNIT

_LENGTH_CONVERSIONS = {"in": 39.37007874015748, "mm": 1000., "cm": 100.}
_AREA_CONVERSIONS = {"in2": 1550.0031000062002, "mm2": 1.0e6, "cm2": 1.0e4}
_ANGLE_CONVERSIONS = {"deg": 0.017453292519943295}


def current_length_unit():
    return _LENGTH_UNIT


def set_current_length_unit(unit: str):
    global _LENGTH_UNIT
    _LENGTH_UNIT = unit


def current_area_unit():
    return _AREA_UNIT


def set_current_area_unit(unit: str):
    global _AREA_UNIT
    _AREA_UNIT = unit


def current_angle_unit():
    return _ANGLE_UNIT


def set_current_angle_unit(unit: str):
    global _ANGLE_UNIT
    _ANGLE_UNIT = unit
