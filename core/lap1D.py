from .div1D import div1D
from .grad1D import grad1D
import numpy as np


def lap1D(k, m, dx):
    """Computes a m+2 by m+2 one-dimensional mimetic laplacian operator

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells
        dx (float): Step size

    Returns:
        :obj:`ndarray` containing discrete laplacian operator
    """

    return np.dot(div1D(k, m, dx), grad1D(k, m, dx))


if __name__ == '__main__':
    print(lap1D(2, 5, 1))
