"""Proposition 2, corrected compression criterion.

A linear summary of sector k preserves the between-sector information for every deformation iff it determines
the profiled GLS estimates c^T theta_hat_k for every c in the shared space
    C_k = row(J~_k)  intersect  sum_{j != k} row(J~_j),
i.e. the combinations sector k measures that are also measured by the other sectors. The earlier statement
("all of J~_k^T W_k must lie in the row space of L_k") is sufficient but not necessary.
Independent implementation: information computed from the compressed Gaussian model directly.
"""
import numpy as np
import pytest


def info_eps(J, G, N):
    """Fisher information on eps with theta profiled, for y = J theta + G eps + n, n ~ N(0, N)."""
    W = np.linalg.inv(N)
    F = np.block([[J.T @ W @ J, J.T @ W @ G], [G.T @ W @ J, G.T @ W @ G]])
    p = J.shape[1]
    Ftt, Fte, Fee = F[:p, :p], F[:p, p:], F[p:, p:]
    return Fee - Fte.T @ np.linalg.pinv(Ftt) @ Fte


def between_info(Js, Gs, Ns, Ls=None):
    """I_b = I(common theta) - I(separate theta_k), from (possibly compressed) data L_k y_k."""
    if Ls is None:
        Ls = [np.eye(J.shape[0]) for J in Js]
    Jz = [L @ J for L, J in zip(Ls, Js)]
    Gz = [L @ G for L, G in zip(Ls, Gs)]
    Nz = [L @ N @ L.T for L, N in zip(Ls, Ns)]
    from scipy.linalg import block_diag
    N = block_diag(*Nz); G = np.vstack(Gz)
    I_common = info_eps(np.vstack(Jz), G, N)
    I_sep = info_eps(block_diag(*Jz), G, N)
    return I_common - I_sep


def make_problem(rng):
    p = 3
    e = np.eye(p)
    J1 = rng.normal(size=(7, p))                                   # sector 1 measures theta_1..3
    J2 = rng.normal(size=(5, 1)) @ e[[0]]                          # sector 2 measures theta_1 only
    J3 = rng.normal(size=(5, 1)) @ e[[1]]                          # sector 3 measures theta_2 only
    Js = [J1, J2, J3]
    Ns = []
    for J in Js:
        A = rng.normal(size=(J.shape[0], J.shape[0]))
        Ns.append(A @ A.T + J.shape[0] * np.eye(J.shape[0]))
    return Js, Ns


def gls_rows(J, N, C):
    """Rows of the linear map y -> C theta_hat (GLS) for one sector."""
    W = np.linalg.inv(N)
    return C @ np.linalg.inv(J.T @ W @ J) @ J.T @ W


@pytest.mark.parametrize("seed", range(5))
def test_dropping_unshared_combination_is_lossless(seed):
    rng = np.random.default_rng(seed)
    Js, Ns = make_problem(rng)
    L1 = gls_rows(Js[0], Ns[0], np.eye(3)[[0, 1]])                 # keep only theta_1, theta_2 estimates
    Ls = [L1, np.eye(5), np.eye(5)]
    # old criterion: rows of J1^T W1 (rank 3) in row space of L1 (rank 2) -> fails
    W1 = np.linalg.inv(Ns[0])
    M = np.vstack([L1, Js[0].T @ W1])
    assert np.linalg.matrix_rank(M) > np.linalg.matrix_rank(L1)
    for _ in range(20):                                            # yet I_b is preserved for every deformation
        Gs = [rng.normal(size=(J.shape[0], 2)) for J in Js]
        assert np.allclose(between_info(Js, Gs, Ns, Ls), between_info(Js, Gs, Ns), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_dropping_shared_combination_loses_information(seed):
    rng = np.random.default_rng(seed)
    Js, Ns = make_problem(rng)
    L1 = gls_rows(Js[0], Ns[0], np.eye(3)[[0, 2]])                 # drop theta_2, which sector 3 also measures
    Ls = [L1, np.eye(5), np.eye(5)]
    W1 = np.linalg.inv(Ns[0])
    a = np.linalg.solve(Js[0].T @ W1 @ Js[0], np.eye(3)[1])        # F1 a = e_2 (shared direction)
    G1 = (Js[0] @ a)[:, None]                                     # deformation biasing the dropped direction
    Gs = [G1, np.zeros((5, 1)), np.zeros((5, 1))]
    full, comp = between_info(Js, Gs, Ns), between_info(Js, Gs, Ns, Ls)
    assert full[0, 0] > 1e-6 and comp[0, 0] < full[0, 0] * (1 - 1e-6)
