from datetime import datetime

from pymead.plugins.IGES.iges_param import IGESParam
from pymead.plugins.IGES import global_section_col_width
from pymead.version import __version__


class GlobalParams:
    """Global parameter section setup for the IGES file format containing defaults for each value"""

    units_indicators = {
        "inches": (1, "INCH"),
        "millimeters": (2, "MM"),
        "feet": (4, "FT"),
        "miles": (5, "MI"),
        "meters": (6, "M"),
        "kilometers": (7, "KM"),
        "mils": (8, "MIL"),
        "microns": (9, "UM"),
        "centimeters": (10, "CM"),
        "microinches": (11, "UIN"),
    }

    def __init__(self):
        self.parameter_delimiter_char = IGESParam(",", "string")
        self.record_delimiter_char = IGESParam(";", "string")
        self.product_id_sender = IGESParam("bezier_curves", "string")
        self.file_name = IGESParam("", "none")
        self._native_system_id = IGESParam(f"pymead {__version__}", "string")
        self._preprocessor_version = IGESParam(f"pymead {__version__}", "string")
        self.binary_bits_int = IGESParam(32, "int")
        self.max_power_sp = IGESParam(38, "int")  # single-precision
        self.sig_digits_sp = IGESParam(16, "int")  # single-precision
        self.max_power_dp = IGESParam(38, "int")  # double-precision
        self.sig_digits_dp = IGESParam(16, "int")  # double-precision
        self.product_id_receiver = IGESParam("bezier_curves", "string")
        self.model_space_scale = IGESParam(1.0, "real")
        self.units_flag = IGESParam(self.units_indicators["millimeters"][0], "int")
        self.units_name = IGESParam(self.units_indicators["millimeters"][1], "string")
        self.max_line_weight_gradations = IGESParam(300, "int")
        self.width_max_line_weight = IGESParam(2.0, "real")
        self.date_and_time_file_generation = IGESParam(datetime.now(), "datetime")
        self.min_model_resolution = IGESParam(1.0e-5, "real")
        self.approx_max_coord_value = IGESParam(100000.0, "real")
        self.author_name = IGESParam("", "none")  # dtype can also be set to "string"
        self.author_org = IGESParam("", "none")  # dtype can also be set to "string"
        self.spec_compliance_flag = IGESParam(11, "int")
        self.spec_drafting_flag = IGESParam(0, "int")
        self.date_time_last_modification = IGESParam(datetime.now(), "datetime")

    def write_globals_string(self):
        gp_list = [v.write_value_to_python_str() for v in vars(self).values()]
        gp_string = self.parameter_delimiter_char.value.join(gp_list)
        gp_string += self.record_delimiter_char.value
        return self.word_wrap_globals(gp_string)

    @staticmethod
    def word_wrap_globals(old_gp_string: str):
        new_gp_string = ""
        start_idx, end_idx = 0, min(global_section_col_width, len(old_gp_string))
        gp_line_counter = 0
        for wrap_idx in range(int(len(old_gp_string) / global_section_col_width) + 1):
            gp_line_counter += 1
            new_gp_string += old_gp_string[start_idx:end_idx]
            new_gp_string += " " * (global_section_col_width - (end_idx - start_idx))
            new_gp_string += f"G{gp_line_counter:7d}\n"
            start_idx += global_section_col_width
            end_idx += min(global_section_col_width,
                           len(old_gp_string) - (wrap_idx + 1) * global_section_col_width)
        return new_gp_string
