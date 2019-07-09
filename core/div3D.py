import numpy as np
from .div1D import div1D
from scipy import sparse
from scipy.sparse import csr_matrix


def div3D(k, m, dx, n, dy, o, dz):
    """Computes a three-dimensional mimetic divergence operator

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells along x-axis
        dx (float): Step size along x-axis
        n (int): Number of cells along y-axis
        dy (float): Step size along y-axis
        o (int): Number of cells along z-axis
        dz (float): Step size along z-axis

    Returns:
        :obj:`ndarray` containing discrete divergence operator
    """

    Im = csr_matrix((m + 2, m), dtype=np.float)
    In = csr_matrix((n + 2, n), dtype=np.float)
    Io = csr_matrix((o + 2, o), dtype=np.float)

    Im[1:m+1, :] = sparse.eye(m, m, dtype=np.float, format='csr')
    In[1:n+1, :] = sparse.eye(n, n, dtype=np.float, format='csr')
    Io[1:o+1, :] = sparse.eye(o, o, dtype=np.float, format='csr')

    Dx = div1D(k, m, dx)
    Dy = div1D(k, n, dy)
    Dz = div1D(k, o, dz)

    Sx = sparse.kron(sparse.kron(Io, In), Dx)
    Sy = sparse.kron(sparse.kron(Io, Dy), Im)
    Sz = sparse.kron(sparse.kron(Dz, In), Im)

    return sparse.hstack([Sx, Sy, Sz], format='csr')


if __name__ == '__main__':
    print(div3D(2, 5, 1, 6, 1, 7, 1))
