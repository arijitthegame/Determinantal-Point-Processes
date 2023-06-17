import numpy as np


def spectral_decomp_sym(B):
    # An input matrix is assumed that B @ B.T.
    eig_vals, eig_vecs_dual = np.linalg.eigh(B.T @ B)
    eig_vecs = (B @ eig_vecs_dual) / (np.sqrt(eig_vals))
    return eig_vals, eig_vecs


# Youla decomposition of B @ C @ B.T where C is skew-symmetric.
def spectral_decomp_skew_sym(B, C):
    assert B.shape[1] == C.shape[0]
    assert type(B) == type(C)
    # note this only works for even dimensional skew-symmetric matrices
    assert divmod(C.shape[0], 2)[-1] == 0
    assert np.allclose(np.linalg.norm(C + C.T, ord='fro'), 1e-10)

    _, k = B.shape

    eig_vals, eig_vecs_dual = np.linalg.eig(C @ (B.T @ B))
    rot = np.kron(1 / np.sqrt(2.0) * np.eye(k // 2), np.array([[1, 1j], [1j, 1]]))

    eig_vecs_unnorm = ((B @ eig_vecs_dual) @ rot).real
    eig_vecs = eig_vecs_unnorm / np.sqrt(np.sum(eig_vecs_unnorm**2, axis=0))
    eig_vals_skew = (rot.conj().T @ np.diag(eig_vals) @ rot).real

    return eig_vals_skew, eig_vecs