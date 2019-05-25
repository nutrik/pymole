import numpy as np
from scipy import sparse
from robinBC import robinBC


def robinBC3D(k, m, dx, n, dy, o, dz, a, b):
    """Computes a three-dimensional mimetic boundary operator that
    imposes a boundary condition of Robin's type

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells along x-axis
        dx (float): Step size along x-axis
        n (int): Number of cells along y-axis
        dy (float): Step size along y-axis
        o (int): Number of cells along z-axis
        dz (float): Step size along z-axis
        a (float): Dirichlet Coefficient
        b (float): Neumann Coefficient

    Returns:
        :obj:`ndarray` containing discrete boundary operator of Robin's type
    """

    """
    1-D boundary operators along each dimension
    """
    Bm = robinBC(k, m, dx, a, b)
    Bn = robinBC(k, n, dy, a, b)
    Bo = robinBC(k, o, dz, a, b)

    Im = sparse.eye(m+2, dtype=np.float, format='csr')

    In = sparse.eye(n+2, dtype=np.float, format='csr')

    Io = sparse.eye(o+2, dtype=np.float, format='csr')
    Io[0, 0] = 0.
    Io[-1, -1] = 0.

    In2 = In.copy()
    In2[0, 0] = 0.
    In2[-1, -1] = 0.

    BC1 = sparse.kron(sparse.kron(Io, In2, format='csr'), Bm, format='csr')
    BC2 = sparse.kron(sparse.kron(Io, Bn, format='csr'), Im, format='csr')
    BC3 = sparse.kron(sparse.kron(Bo, In, format='csr'), Im, format='csr')

    return BC1 + BC2 + BC3


if __name__ == '__main__':
    print(robinBC3D(2, 5, 1, 6, 1, 7, 1, 5, 10).shape)
