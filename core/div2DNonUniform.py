import numpy as np
from .divNonUniform import divNonUniform
from scipy import sparse
from scipy.sparse import csr_matrix


def div2DNonUniform(k, xticks, yticks):
    """Computes a two-dimensional non-uniform mimetic divergence operator

    Arguments:
        k (int): Order of accuracy
        xticks (:obj:`ndarray`): Edges' ticks (x-axis)
        yticks (:obj:`ndarray`): Edges' ticks (y-axis)

    Returns:
        :obj:`ndarray` containing discrete divergence operator
    """

    Dx = divNonUniform(k, xticks)
    Dy = divNonUniform(k, yticks)

    m = Dx.shape[0] - 2
    n = Dy.shape[0] - 2

    Im = csr_matrix((m + 2, m), dtype=np.float)
    In = csr_matrix((n + 2, n), dtype=np.float)

    Im[1:m+1, :] = sparse.eye(m, m, dtype=np.float, format='csr')
    In[1:n+1, :] = sparse.eye(n, n, dtype=np.float, format='csr')

    Sx = sparse.kron(In, Dx, format='csr')
    Sy = sparse.kron(Dy, Im, format='csr')

    return sparse.hstack([Sx, Sy], format='csr')


if __name__ == '__main__':
    xticks = np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45])
    yticks = np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5])
    print(div2DNonUniform(2, xticks, yticks))
    # print(div2DNonUniform(4, xticks, yticks))
    # print(div2DNonUniform(6, xticks, yticks))
