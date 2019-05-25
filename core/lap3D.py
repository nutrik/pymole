import numpy as np
from .div3D import div3D
from .grad3D import grad3D


def lap3D(k, m, dx, n, dy, o, dz):
    """Computes a three-dimensional mimetic laplacian operator

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells along x-axis
        dx (float): Step size along x-axis
        n (int): Number of cells along y-axis
        dy (float): Step size along y-axis
        o (int): Number of cells along z-axis
        dz (float): Step size along z-axis

    Returns:
        :obj:`ndarray` containing discrete laplacian operator
    """

    D = div3D(k, m, dx, n, dy, o, dz)
    G = grad3D(k, m, dx, n, dy, o, dz)

    return np.dot(D, G)


if __name__ == '__main__':
    print(lap3D(2, 5, 1, 6, 1, 7, 1))
