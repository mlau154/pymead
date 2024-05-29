import unittest

class Test(unittest.TestCase):
    def test_one(self):
        self.assertAlmostEqual(2+2, 4)
