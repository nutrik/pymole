import numpy as np
from .div import div
from scipy.sparse import spdiags


def divNonUniform(k, ticks, dx=1.):
    """Computes a m+2 by m+1 one-dimensional non-uniform mimetic divergence
    operator

   Arguments:
        k (int): Order of accuracy
        ticks (:obj:`ndarray`): Edges' ticks e.g. [0 0.1 0.15 0.2 0.3 0.4 0.45]
        dx (float): Grid step size

    Returns:
        :obj:`ndarray` containing discrete divergence operator
    """

    """
    Get uniform operator without scaling
    """
    D = div(k, ticks.size-1, dx)

    m = D.shape[0]

    """
    Compute the Jacobian using the uniform operator and the ticks
    """
    J = spdiags(np.power(np.dot(D.toarray(), ticks), -1), 0, m, m)

    return np.dot(J, D)


if __name__ == '__main__':
    print(divNonUniform(2, np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45]), 1))
