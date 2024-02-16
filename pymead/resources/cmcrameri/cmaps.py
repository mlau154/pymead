import os

import numpy as np

from pymead import RESOURCE_DIR

BERLIN = np.loadtxt(os.path.join(RESOURCE_DIR, "cmcrameri", "berlin.txt"))
VIK = np.loadtxt(os.path.join(RESOURCE_DIR, "cmcrameri", "vik.txt"))
