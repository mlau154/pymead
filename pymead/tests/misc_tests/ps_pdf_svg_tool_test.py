import unittest

from pymead import RESOURCE_DIR
from pymead.utils.file_conversion import convert_ps_to_svg, convert_pdf_to_svg, convert_ps_to_pdf


class VectorGraphicsToolsTest(unittest.TestCase):

    def test_ps_to_svg(self):
        success, logs = convert_ps_to_svg(RESOURCE_DIR, 'plot.ps', 'grid_test.pdf', 'grid_test.svg')
        self.assertTrue(success)

    def test_ps_to_pdf(self):
        success, logs = convert_ps_to_pdf(RESOURCE_DIR, 'plot.ps', 'grid_test.pdf')
        self.assertTrue(success)

    def test_pdf_to_svg(self):
        success, logs = convert_pdf_to_svg(RESOURCE_DIR, 'grid_test.pdf', 'grid_test.svg')
        self.assertTrue(success)
