from pymead.utils.misc import convert_rgba_to_hex


def test_convert_rgba_to_hex():
    rgba = (148, 0, 2, 255)
    hex_out = convert_rgba_to_hex(rgba)
    assert hex_out == "#940002ff"
