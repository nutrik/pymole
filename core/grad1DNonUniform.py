import numpy as np
from .grad1D import grad1D
from scipy.sparse import spdiags


def grad1DNonUniform(k, ticks, dx=1.):
    """ Computes a m+1 by m+2 one-dimensional non-uniform mimetic gradient
    operator

    Arguments:
        k (int): Order of accuracy
        ticks (:obj:`ndarray`): Edges' ticks e.g. [0 0.1 0.15 0.2 0.3 0.4 0.45]
                                (including the boundaries!)
        dx (float): Step size

    Returns:
        :obj:`ndarray` containing discrete gradient operator
    """

    """
    Get uniform operator without scaling
    """
    G = grad(k, ticks.size-2, dx)

    m = G.shape[0]

    """
    Compute the Jacobian using the uniform operator and the ticks
    """
    J = spdiags(np.power(np.dot(G.toarray(), ticks), -1), 0, m, m)

    return np.dot(J, G)


if __name__ == '__main__':
    print(grad1DNonUniform(2,
          np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45]), 1))
    print(grad1DNonUniform(4,
          np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.65]), 1))
