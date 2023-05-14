import os.path
import typing

from pymead.plugins.IGES.start_end_section import StartSection, EndSection
from pymead.plugins.IGES.global_params import GlobalParams
from pymead.plugins.IGES.entity import Entity, MultiEntityContainer


class IGESGenerator:
    """Generates IGES files using a list of IGES entities"""
    def __init__(self, entities: typing.List[Entity]):
        self.entities = entities
        self.start = StartSection()
        self.globals = GlobalParams()
        self.entity_container = MultiEntityContainer(self.entities)
        self.end_section = None

    def generate(self, file_name: str):
        """
        Generates an IGES file containing all the information for the entities.

        Parameters
        ==========
        file_name: str
          File where the IGES data will be saved. If the file name does not end with the ".igs" or ".iges" extension,
          it will be added automatically.

        Returns
        =======
        str
          The IGES data in Python string format
        """

        # First, make sure that the entities know which delimiters to use:
        for entity in self.entity_container.entities:
            entity.param_delimiter = self.globals.parameter_delimiter_char.value
            entity.record_delimiter = self.globals.record_delimiter_char.value

        # Write all the section strings:
        start_section_string = self.start.write_start_section_string()
        global_section_string = self.globals.write_globals_string()
        entity_section_string, data_section_string = self.entity_container.write_all_entity_and_data_strings()
        self.end_section = EndSection(n_start_lines=start_section_string.count("\n"),
                                      n_global_lines=global_section_string.count("\n"),
                                      n_entity_lines=entity_section_string.count("\n"),
                                      n_data_lines=data_section_string.count("\n"))
        end_section_string = self.end_section.write_end_section_string()

        # Add all the section strings together:
        iges_string = start_section_string + global_section_string + entity_section_string + \
                      data_section_string + end_section_string

        # If the file name does not end in the .igs or .iges extension, add the extension:
        if os.path.splitext(file_name)[-1] not in [".igs", ".iges"]:
            file_name += ".igs"

        # Write the total string to the IGES file:
        with open(file_name, "w") as f:
            f.write(iges_string)

        return iges_string
