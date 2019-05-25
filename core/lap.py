from .div import div
from .grad import grad
import numpy as np


def lap(k, m, dx):
    """Computes a m+2 by m+2 one-dimensional mimetic laplacian operator

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells
        dx (float): Step size

    Returns:
        :obj:`ndarray` containing discrete laplacian operator
    """

    return np.dot(div(k, m, dx), grad(k, m, dx))


if __name__ == '__main__':
    print(lap(2, 5, 1))
