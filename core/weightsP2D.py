import numpy as np
from scipy import sparse
from .weightsP1D import weightsP1D


def weightsP2D(k, m, dx, n, dy):
    """Computes the 2mn+m+n weights of P in 2-D

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells along x-axis
        dx (float): Step size along x-axis
        n (int): Number of cells along y-axis
        dy (float): Step size along y-axis

    Returns:
        :obj:`ndarray` containing weights of P
    """

    Im = sparse.eye(m, dtype=np.float, format='csr')
    In = sparse.eye(n, dtype=np.float, format='csr')

    Pm = np.diag(weightsP1D(k, m, dx))
    Pn = np.diag(weightsP1D(k, n, dy))

    return np.concatenate((sparse.kron(In, Pm, format='csr').diagonal(),
                           sparse.kron(Pn, Im, format='csr').diagonal()))


if __name__ == '__main__':
    print(weightsP2D(2, 5, 1, 6, 1).shape)
