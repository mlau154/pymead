import typing


def get_prefix_and_index_from_string(s: str):
    prefix = ''
    idx_str = ''
    for ch in s:
        if ch.isdigit():
            idx_str += ch  # String representation of the FreePoint or AnchorPoint index
        elif ch.isalpha():
            prefix += ch  # Final result of the addition will be 'FP' for a FreePoint or 'AP' for an AnchorPoint
        else:
            raise ValueError(f"Invalid (non alpha-numeric) character found in string {s}")
    idx = int(idx_str)  # Integer representation of the FreePoint or AnchorPoint index
    return prefix, idx


def decrement_string_index(s: str):
    # Get integer index from string
    prefix, idx = get_prefix_and_index_from_string(s)
    # Return a decremented version of the string index
    return prefix + str(idx - 1)


def increment_string_index(s: str):
    # Get integer index from string
    prefix, idx = get_prefix_and_index_from_string(s)
    # Return an incremented version of the string index
    return prefix + str(idx + 1)


def max_string_index_plus_one(str_list: typing.List[str]):
    if str_list and len(str_list) > 0:
        # print(f"{str_list = }")
        idx_list = [int([ch for ch in s if ch.isnumeric()][0]) for s in str_list]
        return 'FP' + str(max(idx_list) + 1)
    else:
        return 'FP0'
