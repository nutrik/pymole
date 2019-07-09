import numpy as np
from .div1D import div1D
from .grad1D import grad1D
from .weightsP1D import weightsP1D
from .weightsQ1D import weightsQ1D
from scipy.sparse import csr_matrix


def mimeticB1D(k, m, dx):
    """Computes a m+2 by m+1 one-dimensional mimetic boundary operator

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells
        dx (float): Step size

    Returns:
        :obj:`ndarray` containing discrete boundary operator
    """

    Q = csr_matrix(np.diag(weightsQ1D(k, m, dx)), dtype=np.float)
    P = csr_matrix(np.diag(weightsP1D(k, m, dx)), dtype=np.float)

    D = div1D(k, m, dx)
    G = grad1D(k, m, dx)

    return np.dot(Q, D) + np.dot(G.T, P)


if __name__ == '__main__':
    print(mimeticB1D(4, 9, 1))
