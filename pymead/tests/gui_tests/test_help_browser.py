import time

from pymead.tests.gui_tests.utils import app


def test_help_browser(app):

    load_successful = False

    def dialog_test_action(dialog):
        dialog.help_browser_widget.help_browser.sigLoadStatusEmitted.connect(on_finished_loading)
        dialog.show()

    def on_finished_loading(ok: bool):
        nonlocal load_successful
        load_successful = ok
        assert ok

    app.show_help(dialog_test_action=dialog_test_action)
    timeout = 3.0  # Timeout in seconds
    must_end = time.time() + timeout
    while time.time() < must_end:
        if load_successful:
            print("load successful")
            break
        time.sleep(0.5)
    if load_successful:
        return
    raise TimeoutError("Could not load the page in time")
