from PyQt6.QtCore import QEvent, QCoreApplication
from PyQt6.QtWidgets import QMenu
from pyqtgraph.parametertree.ParameterItem import ParameterItem

translate = QCoreApplication.translate


def custom_context_menu_event(ev: QEvent, param_item: ParameterItem):
    """
    Re-implemented to provide style inherited from GUI main window
    """
    opts = param_item.param.opts

    if not opts.get('removable', False) and not opts.get('renamable', False) \
            and "context" not in opts:
        return

    ## Generate context menu for renaming/removing parameter

    # Loop to identify the GUI object
    max_iter = 12
    iter_count = 0
    current_parent = param_item.param
    gui = None

    while True:
        if iter_count > max_iter:
            break
        current_parent = current_parent.parent()
        if hasattr(current_parent, "status_bar"):
            gui = current_parent.status_bar.parent()
            break

    ss = gui.themes[gui.current_theme]
    param_item.contextMenu = QMenu(parent=gui)  # Put in global name space to prevent garbage collection
    param_item.contextMenu.setStyleSheet(f"QMenu::item:selected {{ color: {ss['menu-main-color']}; background-color: {ss['menu-item-selected-color']} }} ")
    param_item.contextMenu.addSeparator()
    if opts.get('renamable', False):
        param_item.contextMenu.addAction(translate("ParameterItem", 'Rename')).triggered.connect(param_item.editName)
    if opts.get('removable', False):
        param_item.contextMenu.addAction(translate("ParameterItem", "Remove")).triggered.connect(param_item.requestRemove)

    # context menu
    context = opts.get('context', None)
    if isinstance(context, list):
        for name in context:
            param_item.contextMenu.addAction(name).triggered.connect(
                param_item.contextMenuTriggered(name))
    elif isinstance(context, dict):
        for name, title in context.items():
            param_item.contextMenu.addAction(title).triggered.connect(
                param_item.contextMenuTriggered(name))

    param_item.contextMenu.popup(ev.globalPos())
