import typing

from pymead.plugins.IGES.iges_param import IGESParam


class Entity:

    line_fonts = {
        "no_pattern": 0,
        "solid": 1,
        "dashed": 2,
        "phantom": 3,
        "centerline": 4,
        "dotted": 5,
    }

    color_numbers = {
        "no_color": 0,
        "black": 1,
        "red": 2,
        "green": 3,
        "blue": 4,
        "yellow": 5,
        "magenta": 6,
        "cyan": 7,
        "white": 8,
    }

    def __init__(self, ID: int, parameter_data: typing.List[IGESParam]):
        self.entity_ID = IGESParam(ID, "int")
        self.pd_pointer = IGESParam(1, "int")  # First parameter data line for this entity
        self.structure = IGESParam("", "none")
        self.line_font_pattern = IGESParam(self.line_fonts['solid'], "int")
        self.level = IGESParam(1, "int")
        self.view = IGESParam(0, "int")
        self.transformation_matrix_pointer = IGESParam(0, "int")
        self.label_display_associativity = IGESParam("", "none")
        self.status_number = IGESParam(0, "int")
        self.line_weight_number = IGESParam(105, "int")
        self.color_number = IGESParam(self.color_numbers['cyan'], "int")
        self.parameter_line_count = IGESParam(2, "int")
        self.form_number = IGESParam(0, "int")
        self.reserved = IGESParam("", "none")
        self.entity_label = IGESParam("", "none")
        self.subscript_number = IGESParam(0, "int")
        self.parameter_data = parameter_data
        self.param_delimiter = None
        self.record_delimiter = None

    def write_entity_string(self, entity_starting_line: int, data_starting_line: int, data_string_lines: int):

        entity_string = ""

        def write_line_string(line: typing.List[IGESParam], string_to_write: str):
            for iges_param in line:
                if iges_param.dtype == "none":
                    string_to_write += " " * 8
                elif iges_param.dtype == "int":
                    string_to_write += f"{iges_param.value:8d}"
                else:
                    raise TypeError(f"For an entity, every parameter must have type 'int' or type 'none'. Found an"
                                    f"IGESParam with type {type(iges_param)}.")
            return string_to_write

        self.pd_pointer.value = data_starting_line
        self.parameter_line_count.value = data_string_lines

        line1 = [self.entity_ID, self.pd_pointer, self.structure, self.line_font_pattern, self.level, self.view,
                 self.transformation_matrix_pointer, self.label_display_associativity, self.status_number]
        line2 = [self.entity_ID, self.line_weight_number, self.color_number, self.parameter_line_count,
                 self.form_number, self.reserved, self.reserved, self.entity_label, self.subscript_number]

        entity_string = write_line_string(line1, entity_string) + f"D{entity_starting_line:7d}\n"

        entity_string = write_line_string(line2, entity_string) + f"D{entity_starting_line + 1:7d}\n"

        return entity_string

    def write_data_string(self, entity_entry_line: int, data_starting_line: int):

        data_string = ""

        current_line = self.entity_ID.write_value_to_python_str() + self.param_delimiter

        for p_idx, p in enumerate(self.parameter_data):
            p_str = p.write_value_to_python_str()
            if len(current_line) + len(p_str) < 64:
                current_line += p_str
                if p_idx < len(self.parameter_data) - 1:
                    current_line += self.param_delimiter
                else:
                    current_line += self.record_delimiter
                    current_line += " " * (64 - len(current_line))
                    current_line += f"{entity_entry_line:8d}P{data_starting_line:7d}\n"
                    data_string += current_line
            else:
                current_line += " " * (64 - len(current_line))
                current_line += f"{entity_entry_line:8d}P{data_starting_line:7d}\n"
                data_starting_line += 1
                data_string += current_line
                current_line = p_str + self.param_delimiter

        return data_string


class MultiEntityContainer:
    def __init__(self, entities: typing.List[Entity]):
        self.entities = entities

    def write_all_entity_and_data_strings(self):

        full_entity_string = ""
        full_data_string = ""
        data_starting_lines = [1]
        data_string_lengths = []

        # First pass loop to generate the data strings and the data string line numbers
        for entity_idx, entity in enumerate(self.entities):
            entity_entry_line = 1 + 2 * entity_idx
            data_string = entity.write_data_string(entity_entry_line, data_starting_line=data_starting_lines[-1])
            if entity_idx < len(self.entities) - 1:
                data_starting_lines.append(data_starting_lines[-1] + len(data_string))
            data_string_lengths.append(data_string.count('\n'))

            full_data_string += data_string

        # Second pass loop to use the data string line numbers to generate the entity strings
        for entity_idx, entity in enumerate(self.entities):
            entity_entry_line = 1 + 2 * entity_idx
            full_entity_string += entity.write_entity_string(entity_entry_line, data_starting_lines[entity_idx],
                                                             data_string_lines=data_string_lengths[entity_idx])

        return full_entity_string, full_data_string
