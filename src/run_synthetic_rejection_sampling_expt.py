import time
from itertools import combinations
import numpy as np
from scipy.linalg import block_diag

from utils import get_nonuniform_cluster_matrix
from cholesky_sampling import cholesky_based_sampling
from rejection_sampling import decompose_proposal_dpp, rejection_sampling_ndpp
from tree_sampling import construct_tree


def main():
    random_state = 1
    nonuniform = True
    num_samples = 1
    num_clusters = 100

    n = 2**16
    d1 = 100
    d2 = 100
    d = d1 + d2

    print(f"n = {n}, d = {d}")

    np.random.seed(random_state)

    if nonuniform:
        VB = np.random.rand(get_nonuniform_cluster_matrix(n, d, num_clusters))
        V = VB[:, :d1]
        B = VB[:, d1:]
    else:
        V = np.random.rand(n, d1) / np.sqrt(d1)
        B = np.random.rand(n, d2) / np.sqrt(d2)
    D = np.random.rand(d2, d2)
    C = D - D.T

    # Cholesky-based sampling
    #   preprocessing: matrix-muliplication time for converting L-ensemble
    #                  to marginal kernel (O(nd^2) time)
    #   sampling: M of K-by-K matrix multiplications (O(nd^2) time)
    print(f"==================== Cholesky-based sampling ====================")
    tic = time.time()
    VB = np.concatenate((V, B), axis=1)
    CC = block_diag(np.eye(d1), C) 
    CK = CC @ np.linalg.inv(np.eye(d, dtype=V.dtype) + VB.T @ VB @ CC)
    time_inner = time.time() - tic
    print(f"inner matrix computation time : {time_inner:.5f} sec")

    tic = time.time()
    samples_chol = []
    for _ in range(num_samples):
        samples_chol.append(cholesky_based_sampling(VB, CK))
    time_chol = time.time() - tic
    print(f"chol sampling time: {time_chol:.5f} sec")

    # Rejection sampling
    #   preprocessing: spectral-decomposition & binary tree construction (O(nd^2) time)
    #   sampling: rejection sampling via tree-based symmetric DPP sampling
    print(f"=================== Rejection sampling (tree) ====================")
    tic = time.time()
    X_hat, Z_hat, sigmas = decompose_proposal_dpp(V, B, C)
    tree = construct_tree(np.arange(n), Z_hat.T)
    get_det_L = lambda Y: np.linalg.det(V[Y, :] @ V[Y, :].T + B[Y, :] @ C @ B[Y, :].T).item() #fix
    get_det_Lhat = lambda Y: np.linalg.det((Z_hat[Y, :] * X_hat) @ Z_hat[Y, :].T).item() #fix
    time_spec = time.time() - tic
    print(f"proposal construct time: {time_spec:.3f} sec")

    tic = time.time()
    samples_reject = []
    num_rejections = []
    for _ in range(num_samples):
        sample, num_rejects = rejection_sampling_ndpp(tree, Z_hat, X_hat, get_det_L, get_det_Lhat)
        samples_reject.append(sample)
        num_rejections.append(num_rejects)
    time_rjtn = time.time() - tic
    print(f"rejection sampling time : {time_rjtn:.5f} sec")
    print(f"average num rejections : {np.mean(num_rejections)} ({num_samples})")
    print(f"rejection sampling time : {time_rjtn:.5f} sec")


if __name__ == "__main__":
    main()