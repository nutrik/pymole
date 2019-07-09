import numpy as np
import scipy.sparse.linalg as sp
from .div1D import div1D


def weightsQ1D(k, m, dx):
    """Computes the m+2 weights of Q

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells
        dx (float): Step size

    Returns:
        :obj:`ndarray` containing weights of Q
    """

    D = div1D(k, m, dx)

    b = np.append(np.insert(np.zeros((m-1, 1)), 0, -1), 1)  # RHS

    Q, R = np.linalg.qr(np.transpose(D[1:-1, :]).toarray())
    P = np.dot(Q.T, b)
    return np.append(np.insert(np.linalg.solve(R, P), 0, 1), 1)


if __name__ == '__main__':
    print(weightsQ1D(4, 9, 1))
