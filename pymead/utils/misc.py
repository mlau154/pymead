import os
import re


def count_dollar_signs(input_string: str, search_for_character: str):
    counter = 0
    for ch in input_string:
        if ch == search_for_character:
            counter += 1
    return counter


def make_ga_opt_dir(rootdir: str, ga_opt_dir_name: str):
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


if __name__ == '__main__':
    print(count_dollar_signs("$b = $3 + $A0.Anchor", "$"))
