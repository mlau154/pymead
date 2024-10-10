import matplotlib.pyplot as plt
import numpy as np


def wagner_contour(A: np.ndarray, n_theta: int = 150):
    assert A.ndim == 1
    assert len(A) >= 2
    theta = np.linspace(0, np.pi, n_theta)
    x = 0.5 * (1 - np.cos(theta))
    y = -A[0] * np.sin(0.5 * theta)**2 + A[0] / np.pi * (theta + np.sin(theta))
    for A_idx, A_val in enumerate(A[1:]):
        n = A_idx + 1
        y += A_val / np.pi * (np.sin((n + 1) * theta) / (n + 1) + np.sin(n * theta) / n)
    return np.column_stack((x, y))


def main():
    # xy = wagner_contour(np.array([0.10148, 0.019233, 0.0044033, 0.008108]))
    xy = wagner_contour(np.array([0.071049, 0.011098, 0.0051307]))
    xy2 = wagner_contour(np.array([-0.071049, -0.011098, -0.0051307]))
    plt.plot(xy[:, 0], xy[:, 1])
    plt.plot(xy2[:, 0], xy2[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
