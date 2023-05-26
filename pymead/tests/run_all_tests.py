import unittest


def load_tests():
    """
    Loads all tests found in this module and all submodules with names of the form ``test*.py``.

    Returns
    =======
    unittest.TestSuite
        The suite of tests discovered in the ``pymead.tests`` module
    """
    loader = unittest.TestLoader()
    test_suite = loader.discover(".")
    return test_suite


def run_tests(test_suite: unittest.TestSuite):
    """
    Runs all the tests found in the TestSuite.

    Parameters
    ==========
    test_suite: unittest.TestSuite
        The suite of tests discovered in the ``pymead.tests`` module
    """
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)


def main():
    """
    Loads and runs all tests in the ``pymead.tests`` module.
    """
    test_suite = load_tests()
    run_tests(test_suite)


if __name__ == "__main__":
    main()
