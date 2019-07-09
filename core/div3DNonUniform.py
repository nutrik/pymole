import numpy as np
from .div1DNonUniform import div1DNonUniform
from scipy import sparse
from scipy.sparse import csr_matrix


def div3DNonUniform(k, xticks, yticks, zticks):
    """Computes a three-dimensional non-uniform mimetic divergence operator

    Arguments:
        k (int): Order of accuracy
        xticks (:obj:`ndarray`): Edges' ticks (x-axis)
        yticks (:obj:`ndarray`): Edges' ticks (y-axis)
        zticks (:obj:`ndarray`): Edges' ticks (z-axis)

    Returns:
        :obj:`ndarray` containing discrete divergence operator
    """

    Dx = div1DNonUniform(k, xticks)
    Dy = div1DNonUniform(k, yticks)
    Dz = div1DNonUniform(k, zticks)

    m = Dx.shape[0] - 2
    n = Dy.shape[0] - 2
    o = Dz.shape[0] - 2

    Im = csr_matrix((m + 2, m), dtype=np.float)
    In = csr_matrix((n + 2, n), dtype=np.float)
    Io = csr_matrix((o + 2, o), dtype=np.float)

    Im[1:m+1, :] = sparse.eye(m, m, dtype=np.float, format='csr')
    In[1:n+1, :] = sparse.eye(n, n, dtype=np.float, format='csr')
    Io[1:o+1, :] = sparse.eye(o, o, dtype=np.float, format='csr')

    Sx = sparse.kron(sparse.kron(Io, In), Dx)
    Sy = sparse.kron(sparse.kron(Io, Dy), Im)
    Sz = sparse.kron(sparse.kron(Dz, In), Im)

    return sparse.hstack([Sx, Sy, Sz], format='csr')


if __name__ == '__main__':
    xticks = np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45])
    yticks = np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5])
    zticks = np.array([0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.65])
    print(div3DNonUniform(2, xticks, yticks, zticks).shape)
