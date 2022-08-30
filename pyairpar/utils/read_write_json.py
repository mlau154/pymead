import json


def read_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def write_json(data: dict, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f)
