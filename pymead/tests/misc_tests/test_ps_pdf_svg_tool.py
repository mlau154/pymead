import os
import unittest
import warnings

from pymead import TEST_DIR, DependencyNotFoundError
from pymead.utils.file_conversion import (convert_ps_to_svg, convert_ps_to_pdf)


class VectorGraphicsToolsTest(unittest.TestCase):

    def test_ps_to_svg(self):
        conversion_dir = os.path.join(TEST_DIR, "misc_tests", "ps_svg_pdf_conversion")
        try:
            success, logs = convert_ps_to_svg(
                conversion_dir,
                "plot.ps",
                "grid_test.pdf",
                "grid_test.svg"
            )
            self.assertTrue(success)
            self.assertTrue(os.path.exists(os.path.join(conversion_dir, "grid_test.svg")))
        except DependencyNotFoundError as e:
            warnings.warn(str(e))

    def test_ps_to_pdf(self):
        conversion_dir = os.path.join(TEST_DIR, "misc_tests", "ps_svg_pdf_conversion")
        try:
            success, logs = convert_ps_to_pdf(
                conversion_dir,
                "plot.ps",
                "grid_test.pdf"
            )
            self.assertTrue(success)
            self.assertTrue(os.path.exists(os.path.join(conversion_dir, "grid_test.pdf")))
        except DependencyNotFoundError as e:
            warnings.warn(str(e))
