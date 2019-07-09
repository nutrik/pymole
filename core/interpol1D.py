import numpy as np
from scipy.sparse import csr_matrix


def interpol1D(m, c):
    """Computes a m+1 by m+2 one-dimensional interpolator of 2nd-order

    Arguments:
        m (int): Number of cells
        c (float): Left interpolation coeff.

    Returns:
        :obj:`ndarray` containing coefficients of interpolator
    """

    assert m >= 4, "m must be >= 4, given: {}".format(m)
    assert (c >= 0) and (c <= 1), "0 <= c <= 1, given: {}".format(c)

    """
    Dimensions of I
    """
    n_rows = m + 1
    n_cols = m + 2

    I = csr_matrix((n_rows, n_cols), dtype=np.float)

    I[0, 0] = 1.
    I[-1, -1] = 1.

    """
    Average between two continuous cells
    """
    avg = np.array([c, 1.-c])

    j = 1
    for i in range(1, n_rows - 1):
        I[i, j:j+2] = avg
        j = j + 1

    return I


if __name__ == '__main__':
    print(interpol1D(5, 0.5).toarray())
