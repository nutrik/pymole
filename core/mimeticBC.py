import numpy as np
from .div import div
from .grad import grad
from .weightsP import weightsP
from .weightsQ import weightsQ
from scipy.sparse import csr_matrix


def mimeticB(k, m, dx):
    """Computes a m+2 by m+1 one-dimensional mimetic boundary operator

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells
        dx (float): Step size

    Returns:
        :obj:`ndarray` containing discrete boundary operator
    """

    Q = csr_matrix(np.diag(weightsQ(k, m, dx)), dtype=np.float)
    P = csr_matrix(np.diag(weightsP(k, m, dx)), dtype=np.float)

    D = div(k, m, dx)
    G = grad(k, m, dx)

    return np.dot(Q, D) + np.dot(G.T, P)


if __name__ == '__main__':
    print(mimeticB(4, 9, 1))
