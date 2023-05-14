import numpy as np

from pymead.plugins.IGES.curves import BezierIGES
from pymead.plugins.IGES.iges_generator import IGESGenerator


def main():
    file_name = "iges_generation_example.igs"
    P = np.array([
        [0., 0., 0.],
        [100., 50., 80.],
        [-60., 30., -50.],
        [220., -80., 125.],
        [20., 20., 20.]
    ])
    bez = BezierIGES(P)
    iges_generator = IGESGenerator(entities=[bez])
    iges_generator.generate(file_name)


if __name__ == "__main__":
    main()
