

def count_dollar_signs(input_string: str, search_for_character: str):
    counter = 0
    for ch in input_string:
        if ch == search_for_character:
            counter += 1
    return counter


if __name__ == '__main__':
    print(count_dollar_signs("$b = $3 + $A0.Anchor", "$"))
