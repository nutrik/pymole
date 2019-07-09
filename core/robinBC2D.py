import numpy as np
from scipy import sparse
from .robinBC1D import robinBC1D


def robinBC2D(k, m, dx, n, dy, a, b):
    """Returns a two-dimensional mimetic boundary operator that
    imposes a boundary condition of Robin's type

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells along x-axis
        dx (float): Step size along x-axis
        n (int): Number of cells along y-axis
        dy (float): Step size along y-axis
        a (float): Dirichlet Coefficient
        b (float): Neumann Coefficient

    Returns:
        :obj:`ndarray` containing discrete boundary operator of Robin's type
    """

    Bm = robinBC1D(k, m, dx, a, b)
    Bn = robinBC1D(k, n, dy, a, b)

    Im = sparse.eye(m+2, dtype=np.float, format='csr')

    In = sparse.eye(n+2, dtype=np.float, format='csr')
    In[0, 0] = 0.
    In[-1, -1] = 0.

    BC1 = sparse.kron(In, Bm, format='csr')
    BC2 = sparse.kron(Bn, Im, format='csr')

    return BC1 + BC2


if __name__ == '__main__':
    print(robinBC2D(2, 5, 1, 6, 1, 5, 10)[49, :])
