import requests
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np


def extract_data_from_airfoiltools(name: str):
    url = f'http://airfoiltools.com/airfoil/seligdatfile?airfoil={name}'
    data = requests.get(url)
    text = deepcopy(data.text)
    coords_str = text.split('\n')
    xy_str_list = [coord_str.split() for coord_str in coords_str]
    xy_list = []
    for xy_str in xy_str_list[1:-1]:
        xy = [float(xy_str[0]), float(xy_str[1])]
        xy_list.append(xy)
    xy = np.array(xy_list)
    return xy


def main():
    xy = extract_data_from_airfoiltools('n0012-il')
    plt.plot(xy[:, 0], xy[:, 1])
    plt.show()


if __name__ == '__main__':
    main()
