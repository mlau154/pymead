from pytestqt.qtbot import QtBot

from pymead.tests.gui_tests.utils import app


def test_help_browser(app, qtbot: QtBot):

    load_successful = False

    def dialog_test_action(dialog):
        dialog.help_browser_widget.help_browser.sigLoadStatusEmitted.connect(on_finished_loading)
        qtbot.addWidget(dialog)
        dialog.show()

    def on_finished_loading(ok: bool):
        nonlocal load_successful
        load_successful = ok
        assert ok

    app.show_help(dialog_test_action=dialog_test_action)

    def check_loaded():
        assert load_successful

    qtbot.wait_until(check_loaded, timeout=6000)
