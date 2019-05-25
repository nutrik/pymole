import numpy as np
from .div2D import div2D
from .grad2D import grad2D


def lap2D(k, m, dx, n, dy):
    """Computes a two-dimensional mimetic laplacian operator

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells along x-axis
        dx (float): Step size along x-axis
        n (int): Number of cells along y-axis
        dy (float): Step size along y-axis

    Returns:
        :obj:`ndarray` containing discrete laplacian operator
    """

    D = div2D(k, m, dx, n, dy)
    G = grad2D(k, m, dx, n, dy)

    return np.dot(D, G)


if __name__ == '__main__':
    print(lap2D(2, 5, 1, 6, 1).shape)
