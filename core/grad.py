import numpy as np
from scipy.sparse import csr_matrix, identity


def grad(k, m, dx):
    """Computes a m+1 by m+2 one-dimensional mimetic gradient operator

    Arguments:
        k (int): Order of accuracy
        m (int): Number of cells
        dx (float): Step size

    Returns:
        :obj:`ndarray` containing discrete gradient operator
    """

    assert k >= 2, "Wrong order of accuracy: {}".format(k)
    assert k % 2 == 0, "Order of accuracy must be an even number: {}".format(k)
    assert m >= 2 * k, "m must be >= {} for k = {}".format(2 * k, k)

    """
    Dimensions of G
    """
    n_rows = m + 1
    n_cols = m + 2

    G = csr_matrix((n_rows, n_cols), dtype=np.float)

    """
    Fill the middle of G
    """
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

    j = 1
    for i in range(int(k/2), int(n_rows - k/2)):
        G[i, j:j+k] = coeffs
        j = j + 1

    """
    Create A
    """
    p = int(k / 2)
    q = int(k + 1)
    A = csr_matrix((p, q), dtype=np.float)
    # For each row of A
    for i in range(p):
        """
        k+1 points are used for the boundaries
        Shifting the stencil to the right
        """
        neighbors = np.insert(np.arange(0.5 - i, q - i - 0.5), 0, -i)
        V = np.transpose(np.vander(neighbors))
        b = np.zeros((q, 1), dtype=np.float)
        b[q-2, None] = 1.
        coeffs = np.transpose(np.linalg.solve(V, b))
        A[i, 0:q] = coeffs

    """
    Insert A into G (upper-left corner of G)
    """
    if A.count_nonzero() != 0:
        G[0:p, 0:q] = A

    """
    Permutation matrices
    """
    Pp = csr_matrix(np.fliplr(identity(p).toarray()), dtype=np.float)
    Pq = csr_matrix(np.fliplr(identity(q).toarray()), dtype=np.float)

    """
    Construct A' (lower-right corner of G)
    """
    A = -Pp * A * Pq

    """
    Insert A' into G
    """
    if A.count_nonzero() != 0:
        G[n_rows-p:n_rows, n_cols-q:n_cols] = A

    """
    Scale G
    """
    return G / dx


if __name__ == '__main__':
    grad(2, 4, 1)
    print(grad(4, 8, 1))
    grad(6, 12, 1)
