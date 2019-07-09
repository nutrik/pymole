import numpy as np
from scipy import sparse
from .interpolD1D import interpolD1D
from scipy.sparse import csr_matrix


def interpolD2D(m, n, c1, c2):
    """Computes a two-dimensional interpolator of 2nd-order

    Arguments:
        m (int): Number of cells along x-axis
        n (int): Number of cells along y-axis
        c1 (float): Left interpolation coeff.
        c2 (float): Bottom interpolation coeff.

    Returns:
        :obj:`ndarray` containing coefficients of interpolator
    """

    Ix = interpolD1D(m, c1)
    Iy = interpolD1D(n, c2)

    Im = csr_matrix((m + 2, m), dtype=np.float)
    In = csr_matrix((n + 2, n), dtype=np.float)

    Im[1:m+1, :] = sparse.eye(m, m, dtype=np.float, format='csr')
    In[1:n+1, :] = sparse.eye(n, n, dtype=np.float, format='csr')

    Sx = sparse.kron(In, Ix)
    Sy = sparse.kron(Iy, Im)

    return sparse.hstack([Sx, Sy], format='csr')


if __name__ == '__main__':
    print(interpolD2D(5, 6, 0.5, 0.5))
