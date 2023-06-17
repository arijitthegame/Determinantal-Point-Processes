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

# check for bugs 
def spectral_symmetrization(W, return_decompose=True):
    # W shoud be skew-symmetric
    assert np.allclose(np.linalg.norm(W + W.T, ord='fro'), 1e-10)

    e_complex, V_complex = np.linalg.eig(W)
    # Discard zero eigenvalues and corresponding eigenvectors.
    idx = abs(e_complex.imag) > 1e-12
    V_complex = V_complex[:,idx]
    e_complex = e_complex[idx]

    dtype_ = W.dtype
    k = W.shape[0]

    # Eigenvectors of skew-symmetrix matrix are of form (a+ib, a-ib), hence we 
    # transform this into ((a-b)/2, (a+b)/2) by multiplying rotation matrix and 
    # taking real-value part. 
    rot = np.kron(np.eye(len(e_complex)//2), np.array([[1, -1j], [-1j, 1]]))
    V = (V_complex @ rot).real
    E_skew = (np.diag(e_complex)@rot).real
    E_sym_pre = np.abs(np.diag(E_skew, k=1)[::2])
    E_sym = np.diag(np.repeat(E_sym_pre,2,1)) 
    if return_decompose:
        return V @ E_sym @ V.T
    else:
        return V, E_sym