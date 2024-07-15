from pymead.tests.gui_tests.utils import app


def test_load_examples(app):

    example_dict = {}

    def _test_load_example_recursively(d: dict):
        for k, v in d.items():
            pass

    _test_load_example_recursively(example_dict)

    # assert len(app.geo_col.container()["airfoils"]) != 0
