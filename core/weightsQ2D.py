import numpy as np


def weightsQ2D(m, n, d):
    """Computes the (m+2)(n+2) weights of Q in 2-D

    Arguments:
        m (int): Number of cells along x-axis
        n (int): Number of cells along y-axis
        d (float): Step size (assuming d = dx = dy)

    Only works for 2nd-order 2-D Mimetic divergence operator

    Returns:
        :obj:`ndarray` containing weights of Q
    """

    return d * np.ones((m+2)*(n+2), dtype=np.float)


if __name__ == '__main__':
    print(weightsQ2D(5, 6, 1).shape)
