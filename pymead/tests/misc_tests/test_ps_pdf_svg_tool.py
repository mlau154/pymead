import unittest

from pymead import RESOURCE_DIR
from pymead.utils.file_conversion import (convert_ps_to_svg, convert_pdf_to_svg, convert_ps_to_pdf,
                                          DependencyNotFoundError)


class VectorGraphicsToolsTest(unittest.TestCase):

    def test_ps_to_svg(self):
        try:
            success, logs = convert_ps_to_svg(".", 'plot.ps', 'grid_test.pdf', 'grid_test.svg')
            self.assertTrue(success)
        except DependencyNotFoundError as e:
            print(f"Warning: {str(e)}")

    def test_ps_to_pdf(self):
        try:
            success, logs = convert_ps_to_pdf(".", 'plot.ps', 'grid_test.pdf')
            self.assertTrue(success)
        except DependencyNotFoundError as e:
            print(f"Warning: {str(e)}")

    def test_pdf_to_svg(self):
        try:
            success, logs = convert_pdf_to_svg(".", 'grid_test.pdf', 'grid_test.svg')
            self.assertTrue(success)
        except DependencyNotFoundError as e:
            print(f"Warning: {str(e)}")
