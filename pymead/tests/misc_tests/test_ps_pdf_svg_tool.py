import os
import unittest

from pymead import TEST_DIR, DependencyNotFoundError
from pymead.utils.file_conversion import (convert_ps_to_svg, convert_pdf_to_svg, convert_ps_to_pdf)


class VectorGraphicsToolsTest(unittest.TestCase):

    def test_ps_to_svg(self):
        try:
            success, logs = convert_ps_to_svg(
                os.path.join(TEST_DIR, "misc_tests", "ps_svg_pdf_conversion"),
                "plot.ps",
                "grid_test.pdf",
                "grid_test.svg"
            )
            self.assertTrue(success)
        except DependencyNotFoundError as e:
            print(f"Warning: {str(e)}")

    def test_ps_to_pdf(self):
        try:
            success, logs = convert_ps_to_pdf(
                os.path.join(TEST_DIR, "misc_tests", "ps_svg_pdf_conversion"),
                "plot.ps",
                "grid_test.pdf"
            )
            self.assertTrue(success)
        except DependencyNotFoundError as e:
            print(f"Warning: {str(e)}")

    def test_pdf_to_svg(self):
        try:
            success, logs = convert_pdf_to_svg(
                os.path.join(TEST_DIR, "misc_tests", "ps_svg_pdf_conversion"),
                "grid_test.pdf",
                "grid_test.svg"
            )
            self.assertTrue(success)
        except DependencyNotFoundError as e:
            print(f"Warning: {str(e)}")
