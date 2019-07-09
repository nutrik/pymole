import numpy as np
from .div1D import div1D
from scipy import sparse
from scipy.sparse import csr_matrix


def div2D(k, m, dx, n, dy):
    """Computes a two-dimensional mimetic divergence operator

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells
        dx (float): Step size along x-axis
        n (int) : Number of cells along y-axis
        dy (float): Step size along y-axis

    Returns:
        :obj:`ndarray` containing discrete divergence operator
    """

    Dx = div1D(k, m, dx)
    Dy = div1D(k, n, dy)

    Im = csr_matrix((m + 2, m), dtype=np.float)
    In = csr_matrix((n + 2, n), dtype=np.float)

    Im[1:m+1, :] = sparse.eye(m, m, dtype=np.float, format='csr')
    In[1:n+1, :] = sparse.eye(n, n, dtype=np.float, format='csr')

    Sx = sparse.kron(In, Dx, format='csr')
    Sy = sparse.kron(Dy, Im, format='csr')

    return sparse.hstack([Sx, Sy], format='csr')


if __name__ == '__main__':
    print(div2D(2, 5, 1, 5, 1))
    div2D(4, 9, 1)
    div2D(6, 13, 1)
