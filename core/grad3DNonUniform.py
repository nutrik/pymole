import numpy as np
from .grad1DNonUniform import grad1DNonUniform
from scipy import sparse
from scipy.sparse import csr_matrix


def grad3DNonUniform(k, xticks, yticks, zticks):
    """Computes a three-dimensional non-uniform mimetic gradient operator

    Arguments:
        k (int): Order of accuracy
        xticks (:obj:`ndarray`): Centers' ticks (x-axis)
        yticks (:obj:`ndarray`): Centers' ticks (y-axis)
        zticks (:obj:`ndarray`): Centers' ticks (z-axis)
                                 (including the boundaries!)

    Returns:
        :obj:`ndarray` containing discrete gradient operator
    """

    Gx = grad1DNonUniform(k, xticks)
    Gy = grad1DNonUniform(k, yticks)
    Gz = grad1DNonUniform(k, zticks)

    m = Gx.shape[0] - 1
    n = Gy.shape[0] - 1
    o = Gz.shape[0] - 1

    Im = csr_matrix((m + 2, m), dtype=np.float)
    In = csr_matrix((n + 2, n), dtype=np.float)
    Io = csr_matrix((o + 2, o), dtype=np.float)

    Im[1:m+1, :] = sparse.eye(m, m, dtype=np.float, format='csr')
    In[1:n+1, :] = sparse.eye(n, n, dtype=np.float, format='csr')
    Io[1:o+1, :] = sparse.eye(o, o, dtype=np.float, format='csr')

    Sx = sparse.kron(sparse.kron(Io.T, In.T), Gx)
    Sy = sparse.kron(sparse.kron(Io.T, Gy), Im.T)
    Sz = sparse.kron(sparse.kron(Gz, In.T), Im.T)

    return sparse.vstack([Sx, Sy, Sz], format='csr')


if __name__ == '__main__':
    xticks = np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45])
    yticks = np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5])
    zticks = np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.65])
    print(grad3DNonUniform(2, xticks, yticks, zticks).shape)
