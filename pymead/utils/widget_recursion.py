from PyQt5.QtWidgets import QWidget


def get_parent(widget: QWidget, depth: int = 1):
    depth -= 1
    if depth == 0:
        return widget.parent()
    else:
        if widget.parent():
            return get_parent(widget.parent(), depth=depth)
        else:
            return None
