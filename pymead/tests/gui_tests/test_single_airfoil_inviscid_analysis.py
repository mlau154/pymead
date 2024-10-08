from pymead.tests.gui_tests.utils import app


def test_single_airfoil_inviscid_analysis(app):

    airfoil = app.load_example_basic_airfoil_sharp_dv()
    app.permanent_widget.inviscid_cl_combo.setCurrentText("Airfoil-1")

    def dialog_action(dialog):
        dialog.setModal(False)
        dialog.show()
        dialog.w.widget_dict["alfa"].setValue(5.0)
        dialog.accept()

    app.single_airfoil_inviscid_analysis(plot_cp=True, dialog_test_action=dialog_action)


def test_single_airfoil_inviscid_analysis_empty(app):
    def dialog_action_info(dialog):
        dialog.setModal(False)
        dialog.show()
        assert dialog.windowTitle() == "Information"
        dialog.accept()

    app.single_airfoil_inviscid_analysis(plot_cp=True, info_dialog_action=dialog_action_info)

