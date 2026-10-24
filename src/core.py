"""
core.py -- linear-Gaussian framework for multi-sector Kerr consistency tests (v2).

Observable space
----------------
Sector k:  y_k = J_k theta + B_k beta_k + G_k eps + n_k,   n_k ~ N(0, N_k),  W_k = N_k^{-1}
  theta (p)   common Kerr parameters        beta_k  nuisances private to sector k
  eps   (q)   deformation parameters        sectors independent unless a full N is given
Nested model spaces (W inner product):
  S0 = col[J, blockdiag(B_k)]   (one Kerr spacetime, private nuisances)
  S1 = col blockdiag[J_k, B_k]  (each sector its own Kerr parameters)

Summary space
-------------
Each sector profiles its private nuisances and reports the GLS estimate z_k of the r_k
identifiable combinations A_k theta of the *profiled* Jacobian  Jt_k = (1 - H_{B_k}) J_k.

Conventions for degeneracy (v2): a parameter that is not identifiable gets sigma = inf,
never a finite number from a pseudo-inverse; degrees of freedom are ranks, never counts.
"""
import numpy as np

TOL = 1e-10


# ---------------------------------------------------------------- linear algebra helpers
def block_diag(*Ms):
    Ms = [np.atleast_2d(M) for M in Ms]
    n = sum(M.shape[0] for M in Ms); m = sum(M.shape[1] for M in Ms)
    out = np.zeros((n, m)); i = j = 0
    for M in Ms:
        out[i:i + M.shape[0], j:j + M.shape[1]] = M
        i += M.shape[0]; j += M.shape[1]
    return out


def whiten(W):
    """Return S with S^T S = W (so the W inner product becomes Euclidean after x -> S x)."""
    return np.linalg.cholesky(W).T


def orth_basis(X, W, tol=TOL):
    """W-orthonormal basis (columns) of col(X), rank-revealing."""
    if X.size == 0 or X.shape[1] == 0:
        return np.zeros((X.shape[0], 0))
    S = whiten(W)
    U, s, _ = np.linalg.svd(S @ X, full_matrices=False)
    r = int(np.sum(s > tol * max(1.0, s[0]))) if s.size else 0
    return np.linalg.solve(S, U[:, :r])


def rank(X, W=None, tol=TOL):
    if X.size == 0 or X.shape[1] == 0:
        return 0
    W = np.eye(X.shape[0]) if W is None else W
    return orth_basis(X, W, tol).shape[1]


def wproj(X, W, tol=TOL):
    """W-orthogonal projector onto col(X) (rank-revealing)."""
    Q = orth_basis(X, W, tol)
    return Q @ Q.T @ W


# ---------------------------------------------------------------- observable-space split (P1)
def info_split(Js, Gs, Ns, Bs=None, Nfull=None):
    """Fisher information on eps (q x q), marginalised over theta and private nuisances.
    total = within + between exactly (P1).  'within_k' are the per-sector quantities each
    sector can compute alone with its own noise block (equal to the abstract within-term
    only when sectors are independent)."""
    Bs = Bs if Bs is not None else [np.zeros((J.shape[0], 0)) for J in Js]
    J = np.vstack(Js); G = np.vstack(Gs)
    Wblk = block_diag(*[np.linalg.inv(N) for N in Ns])
    W = np.linalg.inv(Nfull) if Nfull is not None else Wblk
    Bblk = block_diag(*Bs) if sum(B.shape[1] for B in Bs) else np.zeros((J.shape[0], 0))
    X0 = np.hstack([J, Bblk])
    X1 = block_diag(*[np.hstack([Jk, Bk]) for Jk, Bk in zip(Js, Bs)])
    H0, H1 = wproj(X0, W), wproj(X1, W)
    I = np.eye(J.shape[0])
    within_k = []
    for Jk, Bk, Gk, Nk in zip(Js, Bs, Gs, Ns):
        Wk = np.linalg.inv(Nk); Hk = wproj(np.hstack([Jk, Bk]), Wk)
        within_k.append(Gk.T @ Wk @ (np.eye(Jk.shape[0]) - Hk) @ Gk)
    return dict(total=G.T @ W @ (I - H0) @ G,
                within_abstract=G.T @ W @ (I - H1) @ G,
                between_abstract=G.T @ W @ (H1 - H0) @ G,
                within=sum(within_k), within_k=within_k,
                nu=rank(X1, W) - rank(X0, W))


# ---------------------------------------------------------------- summaries with profiling (P2)
def summary_from_sectors(Js, Gs, Ns, Bs=None):
    """Per-sector profiled GLS summaries.
    Returns A (stacked identifiable directions), b (bias), Sigmas, Ls (z_k = L_k y_k), ranks."""
    Bs = Bs if Bs is not None else [np.zeros((J.shape[0], 0)) for J in Js]
    As, bs, Ls, Sigs, ranks = [], [], [], [], []
    for Jk, Gk, Nk, Bk in zip(Js, Gs, Ns, Bs):
        Wk = np.linalg.inv(Nk)
        Jt = (np.eye(Jk.shape[0]) - wproj(Bk, Wk)) @ Jk if Bk.shape[1] else Jk
        Ft = Jt.T @ Wk @ Jt
        U, s, Vt = np.linalg.svd(Ft)
        r = int(np.sum(s > TOL * max(1.0, s[0]))) if s.size else 0
        Ak = Vt[:r]
        Ft_pinv = (Vt[:r].T / s[:r]) @ Vt[:r]          # exact inverse on the identifiable subspace
        Lk = Ak @ Ft_pinv @ Jt.T @ Wk
        As.append(Ak); Ls.append(Lk); bs.append(Lk @ Gk); Sigs.append(Lk @ Nk @ Lk.T); ranks.append(r)
    return dict(A=np.vstack(As), b=np.vstack(bs), Ls=Ls, Sigmas=Sigs, ranks=ranks)


def summary_cov(Ls, Nfull, sizes=None):
    L = block_diag(*Ls)
    return L @ Nfull @ L.T


# ---------------------------------------------------------------- summary-space statistics (P3-P5)
def _cols(S):
    if S is None or np.size(S) == 0:
        return None
    S = np.asarray(S, float)
    return S[:, None] if S.ndim == 1 else S


def residual_projector(A, C, S=None):
    """(1 - P) with P the C^{-1}-orthogonal projector onto col[A, S] (S = profiled templates)."""
    Ci = np.linalg.inv(C)
    S = _cols(S)
    X = A if S is None else np.hstack([A, S])
    return np.eye(A.shape[0]) - wproj(X, Ci), Ci


def between_info_summary(A, b, C, S=None):
    R, Ci = residual_projector(A, C, S)
    U = R @ _cols(b)
    return U.T @ Ci @ U


def dof(A, C, S=None):
    S = _cols(S)
    X = A if S is None else np.hstack([A, S])
    return A.shape[0] - rank(X, np.linalg.inv(C))


def decompose_T2(z, A, C, b, S=None):
    """T2 = T2_par + T2_perp with rank-based degrees of freedom.
    q_eff = rank of U = (1-P)b;  nu = dim residual space.  eps not identifiable -> sigma inf."""
    R, Ci = residual_projector(A, C, S)
    r = R @ z
    T2 = float(r @ Ci @ r)
    b2 = b[:, None] if np.ndim(b) == 1 else b
    U = R @ b2
    Q = orth_basis(U, Ci)                         # C^{-1}-orthonormal basis of col(U)
    q_eff = Q.shape[1]
    Tpar = float(np.sum((Q.T @ Ci @ r) ** 2)) if q_eff else 0.0
    nu = dof(A, C, S)
    # estimate and covariance of eps (only if fully identifiable)
    F = U.T @ Ci @ U
    if q_eff < b2.shape[1]:
        eps_hat = np.full(b2.shape[1], np.nan); cov = np.full((b2.shape[1],) * 2, np.inf)
    else:
        cov = np.linalg.inv(F); eps_hat = cov @ (U.T @ Ci @ r)
    return dict(T2=T2, Tpar=Tpar, Tperp=T2 - Tpar, nu=nu, q_eff=q_eff, nu_perp=nu - q_eff,
                eps_hat=eps_hat, cov_eps=cov)


def confusion(A, C, u_vec, s_vec, S=None):
    """|cos phi| between two templates in the residual space. nan if either is blind."""
    R, Ci = residual_projector(A, C, S)
    u = R @ np.asarray(u_vec, float); s = R @ np.asarray(s_vec, float)
    nu_, ns_ = u @ Ci @ u, s @ Ci @ s
    ref_u = np.asarray(u_vec, float) @ Ci @ np.asarray(u_vec, float)
    ref_s = np.asarray(s_vec, float) @ Ci @ np.asarray(s_vec, float)
    if nu_ <= TOL * max(ref_u, 1e-300) or ns_ <= TOL * max(ref_s, 1e-300):
        return np.nan
    return float(min(1.0, abs(u @ Ci @ s) / np.sqrt(nu_ * ns_)))


def sigma_eps(A, C, b, nuisance_templates=()):
    """1-sigma error on one deformation parameter after profiling the common parameters and any
    competing templates. Returns inf when the deformation is not identifiable (P5 with sin phi = 0)."""
    S = np.column_stack([np.asarray(t, float) for t in nuisance_templates]) if len(nuisance_templates) else None
    R, Ci = residual_projector(A, C, S)
    u = R @ np.asarray(b, float)
    info = float(u @ Ci @ u)
    # identifiability threshold relative to the unprojected template norm
    bb = np.asarray(b, float)
    ref = float(bb @ Ci @ bb)
    if ref <= 0 or info <= TOL * ref:
        return np.inf
    return 1.0 / np.sqrt(info)


# ---------------------------------------------------------------- static-metric kernels
def kernel(n, probe, r=None):
    """Linear mass-bias kernels for A(r) = 1 - 2M/r + d_n (M/r)^n, g_rr = 1/A (units M = 1).
    Scope: static, spherically symmetric deformations of Schwarzschild -- not rotating Kerr."""
    if probe == 'orbit':
        return -(n / 2.0) * np.asarray(r, float) ** (1 - n)
    if probe in ('shadow', 'eikonal_wR'):
        return -0.5 * 3.0 ** (1 - n)
    if probe == 'isco_freq':
        return -3.0 * (2 * n * n - n + 1) / (2.0 * 6.0 ** n)     # closed form; -1/2, -7/24, -1/9, -29/864 for n = 1..4
    if probe == 'eikonal_wI':                                       # Lyapunov exponent, linked family A = B
        return (n * n + n - 8) / (4.0 * 3.0 ** n)                   # -1/2, -1/18, 1/27, 1/27, 11/486 for n = 1..5
    if probe == 'grr_wI':                                           # regular radial-only family:
        return -0.5 * 3.0 ** (-n)                                   # A = 1-2M/r, B = (1-2M/r)(1+gamma (M/r)^n)
    if probe in ('grr_orbit', 'grr_isco', 'grr_shadow'):            # these depend on g_tt only
        return 0.0
    raise ValueError(probe)
