import numpy as np
from scipy.sparse import csr_matrix, identity


def div1D(k, m, dx):
    """ Computes a m+2 by m+1 one-dimensional mimetic divergence operator

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells
        dx (float): Grid step size

    Returns:
        :obj:`ndarray` containing discrete divergence operator
    """

    assert k >= 2, "Wrong order of accuracy: {}".format(k)
    assert k % 2 == 0, "Order of accuracy must be an even number: {}".format(k)
    assert m >= 2*k + 1, "m must be >= {} for k = {}".format(2*k + 1, k)

    """
    Dimensions of matrix D
    """
    n_rows = m + 2
    n_cols = m + 1

    D = csr_matrix((n_rows, n_cols), dtype=np.float)

    """
    Fill the middle of D
    """
    # Bandwidth = k
    neighbors = np.arange(0.5 - k/2., k/2. + 0.5)

    """
    Create a k by k Vandermonde matrix based on the neighbors
    """
    A = np.transpose(np.vander(neighbors))

    """
    First-order derivative
    """
    b = np.zeros((k, 1), dtype=np.float)
    b[k-2, None] = 1.

    """
    Solve the linear system to get the coefficients
    """
    coeffs = np.transpose(np.linalg.solve(A, b))

    j = 0
    for i in range(int(k/2), int(n_rows - k/2)):
        D[i, j:j+k] = coeffs
        j = j + 1

    """
    Create A
    """
    p = int(k/2 - 1)
    q = int(k + 1)
    A = csr_matrix((p, q), dtype=np.float)
    # For each row of A
    for i in range(p):
        """
        k+1 points are used for the boundaries
        Shifting the stencil to the right
        """
        neighbors = np.arange(-0.5 - i, q - i - 0.5)
        V = np.transpose(np.vander(neighbors))
        b = np.zeros((q, 1), dtype=np.float)
        b[q-2, None] = 1.
        coeffs = np.transpose(np.linalg.solve(V, b))
        A[i, 0:q] = coeffs

    """
    Insert A into D (upper-left corner of D)
    """
    if A.count_nonzero() != 0:
        D[1:p+1, 0:q] = A
    """
    Permutation matrices
    """
    Pp = csr_matrix(np.fliplr(identity(p).toarray()), dtype=np.float)
    Pq = csr_matrix(np.fliplr(identity(q).toarray()), dtype=np.float)

    """
    Construct A' (lower-right corner of D)
    """
    A = -Pp * A * Pq

    """
    Insert A' into D
    """
    if A.count_nonzero() != 0:
        D[n_rows-p-1:n_rows-1, n_cols-q:n_cols] = A

    """
    Scale D
    """
    return D / dx


if __name__ == '__main__':
    print(div1D(2, 5, 1).toarray())
    print(div1D(4, 9, 1))
    print(div1D(6, 13, 1))
