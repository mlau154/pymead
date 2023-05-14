from pymead.plugins.IGES import global_section_col_width


class StartSection:
    def __init__(self, n_start_lines: int = 1):
        self.n_start_lines = n_start_lines

    def write_start_section_string(self):
        start_section_string = ""
        for start_idx in range(self.n_start_lines):
            start_section_string += " " * global_section_col_width
            start_section_string += f"S{start_idx + 1:7d}\n"
        return start_section_string


class EndSection:
    def __init__(self, n_start_lines, n_global_lines, n_entity_lines, n_data_lines, n_end_lines: int = 1):
        self.n_start_lines = n_start_lines
        self.n_global_lines = n_global_lines
        self.n_entity_lines = n_entity_lines
        self.n_data_lines = n_data_lines
        self.n_end_lines = n_end_lines

    def write_end_section_string(self):
        end_section_string = \
            f"S{self.n_start_lines:7d}G{self.n_global_lines:7d}D{self.n_entity_lines:7d}P{self.n_data_lines:7d}"
        end_section_string += " " * (global_section_col_width - len(end_section_string))
        end_section_string += f"T{self.n_end_lines:7d}\n"
        return end_section_string
