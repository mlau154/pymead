import os
import re

from PyQt6.QtCore import Qt
from pymead import GUI_SETTINGS_DIR, q_settings

from pymead.utils.read_write_files import load_data


qsd = load_data(os.path.join(GUI_SETTINGS_DIR, "q_settings_descriptions.json"))


def count_dollar_signs(input_string: str, search_for_character: str):
    """
    Counts the number of dollar signs in ``input_string``. Useful for counting the number of dynamically-linked
    variables in a user-defined equation, since these are defined by pre-pending the dollar symbol.
    """
    counter = 0
    for ch in input_string:
        if ch == search_for_character:
            counter += 1
    return counter


def count_func_strs(file_name: str):
    """
    Counts the number of 'func_str' in a JMEA file that are not null
    """
    with open(file_name, "r") as f:
        lines = f.readlines()
    non_null_func_str = [True for line in lines if "func_str" in line and "null" not in line]
    return len(non_null_func_str)


def make_ga_opt_dir(rootdir: str, ga_opt_dir_name: str):
    """
    Creates a clean directory for optimization by finding the integer tags of all the directories in the root
    optimization directory with the same name as ``ga_opt_dir_name`` and incrementing the maximum integer by one.
    For example, if the root optimization directory contains subdirectories named ``ga_opt_5``, ``ga_opt_7``,
    and ``ga_opt_8``, the newly created directory will be named ``ga_opt_9``.
    """
    subdir = [folder for folder in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, folder))]
    append_num_list = []
    for folder in subdir:
        if ga_opt_dir_name in folder:
            if len(os.listdir(os.path.join(rootdir, folder))) == 0:
                opt_dir = os.path.join(rootdir, folder)
                return opt_dir
            else:
                append_num_list.append(int(re.split('_', folder)[-1]))
    if len(append_num_list) == 0:
        append_num = 0
    else:
        append_num = max(append_num_list) + 1
    opt_dir = os.path.join(rootdir, f"{ga_opt_dir_name}_{append_num}")
    if not os.path.exists(opt_dir):
        os.mkdir(opt_dir)
    return opt_dir


def convert_str_to_Qt_dash_pattern(dash: str):
    data = {"-": Qt.PenStyle.SolidLine,
            "--": Qt.PenStyle.DashLine,
            ":": Qt.PenStyle.DotLine,
            "-.": Qt.PenStyle.DashDotLine,
            "-..": Qt.PenStyle.DashDotDotLine}
    return data[dash]


def get_setting(key: str):
    return q_settings.value(key, qsd[key][1])


def set_setting(key: str, value: object):
    q_settings.setValue(key, value)
