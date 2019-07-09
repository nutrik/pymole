import numpy as np
import scipy.sparse.linalg as sp
from .grad1D import grad1D


def weightsP1D(k, m, dx):
    """ Computes the m+1 weights of P

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells
        dx (float): Step size

    Returns:
        :obj:`ndarray` containing weights of P
    """

    G = grad1D(k, m, dx)

    b = np.append(np.insert(np.zeros((m, 1)), 0, -1), 1)  # RHS

    Q, R = np.linalg.qr(np.transpose(G).toarray())
    P = np.dot(Q.T, b)
    return np.linalg.solve(R, P)


if __name__ == '__main__':
    print(weightsP1D(6, 12, 1))
