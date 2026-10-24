"""
Independent unit tests for core.py.

The reference implementation here never uses projectors: it fits noiseless ("Asimov") data
y = G eps by weighted least squares with numpy.linalg.lstsq. For eps = 1,
  chi2_min(model S0) = I_total,   chi2_min(model S1) = I_within,   difference = I_between.
"""
import os, sys
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core import (info_split, summary_from_sectors, between_info_summary, block_diag, decompose_T2,
                  sigma_eps, confusion, residual_projector, summary_cov, wproj, dof)

RTOL = 1e-9


# ---------------------------------------------------------------- independent reference
def asimov_chi2(X, y, N):
    """min_x (y - X x)^T N^{-1} (y - X x) by whitened least squares (handles rank deficiency)."""
    Lc = np.linalg.cholesky(N)
    Xw = np.linalg.solve(Lc, X); yw = np.linalg.solve(Lc, y)
    if X.shape[1] == 0:
        return float(yw @ yw)
    sol, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    res = yw - Xw @ sol
    return float(res @ res)


def asimov_split(Js, Gs, Ns, Bs=None, Nfull=None):
    Bs = Bs if Bs is not None else [np.zeros((J.shape[0], 0)) for J in Js]
    N = Nfull if Nfull is not None else block_diag(*Ns)
    y = np.vstack(Gs)[:, 0]
    X0 = np.hstack([np.vstack(Js), block_diag(*Bs) if sum(B.shape[1] for B in Bs) else np.zeros((len(y), 0))])
    X1 = block_diag(*[np.hstack([J, B]) for J, B in zip(Js, Bs)])
    c0, c1 = asimov_chi2(X0, y, N), asimov_chi2(X1, y, N)
    return c0, c1, c0 - c1


def asimov_summary_between(A, b, C):
    """Between information from summaries: chi2 of z = b against model z = A theta."""
    return asimov_chi2(A, b[:, 0] if b.ndim == 2 else b, C)


# ---------------------------------------------------------------- fixtures
def spd(rng, n, cond=5.0):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q @ np.diag(np.exp(rng.uniform(0, np.log(cond), n))) @ Q.T


def random_problem(rng, spec=((5, 2, 1), (3, 1, 0), (4, 2, 1)), p=2, q=1):
    Js, Gs, Ns, Bs = [], [], [], []
    for n, rk, nb in spec:
        Js.append(rng.normal(size=(n, rk)) @ rng.normal(size=(rk, p)))
        Gs.append(rng.normal(size=(n, q))); Ns.append(spd(rng, n)); Bs.append(rng.normal(size=(n, nb)))
    return Js, Gs, Ns, Bs


# ---------------------------------------------------------------- P1 / P2 against the reference
@pytest.mark.parametrize("seed", range(20))
def test_split_matches_asimov(seed):
    rng = np.random.default_rng(seed)
    Js, Gs, Ns, Bs = random_problem(rng)
    s = info_split(Js, Gs, Ns, Bs=Bs)
    c0, c1, cb = asimov_split(Js, Gs, Ns, Bs=Bs)
    assert np.isclose(s['total'].item(), c0, rtol=RTOL, atol=1e-12)
    assert np.isclose(s['within_abstract'].item(), c1, rtol=RTOL, atol=1e-12)
    assert np.isclose(s['within'].item(), c1, rtol=RTOL, atol=1e-12)          # independent sectors
    assert np.isclose(s['between_abstract'].item(), cb, rtol=1e-7, atol=1e-11)


@pytest.mark.parametrize("seed", range(20))
def test_profiled_summary_matches_asimov(seed):
    rng = np.random.default_rng(100 + seed)
    Js, Gs, Ns, Bs = random_problem(rng)
    sm = summary_from_sectors(Js, Gs, Ns, Bs=Bs)
    _, _, cb = asimov_split(Js, Gs, Ns, Bs=Bs)
    Ib = asimov_summary_between(sm['A'], sm['b'], block_diag(*sm['Sigmas']))
    assert np.isclose(Ib, cb, rtol=1e-7, atol=1e-11)


# ---------------------------------------------------------------- exact rank loss
def test_exact_rank_one_sector_dof():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2,))
    J_rank1 = np.outer(rng.normal(size=4), a)            # exactly rank 1 in theta
    Js = [rng.normal(size=(3, 2)), J_rank1, rng.normal(size=(5, 2))]
    Gs = [rng.normal(size=(n, 1)) for n in (3, 4, 5)]; Ns = [spd(rng, n) for n in (3, 4, 5)]
    sm = summary_from_sectors(Js, Gs, Ns)
    assert sm['ranks'] == [2, 1, 2]
    assert dof(sm['A'], block_diag(*sm['Sigmas'])) == 2 + 1 + 2 - 2
    assert info_split(Js, Gs, Ns)['nu'] == 3


def test_common_shift_deformation_is_non_identifiable():
    rng = np.random.default_rng(2)
    Js = [rng.normal(size=(4, 2)) for _ in range(3)]
    c = rng.normal(size=(2, 1))
    Gs = [J @ c for J in Js]                              # deformation = common change of theta
    Ns = [spd(rng, 4) for _ in range(3)]
    s = info_split(Js, Gs, Ns)
    assert abs(s['total'].item()) < 1e-10
    sm = summary_from_sectors(Js, Gs, Ns); C = block_diag(*sm['Sigmas'])
    assert sigma_eps(sm['A'], C, sm['b'][:, 0]) == np.inf
    d = decompose_T2(rng.normal(size=sm['A'].shape[0]), sm['A'], C, sm['b'])
    assert d['q_eff'] == 0 and d['Tpar'] == 0.0 and np.isinf(d['cov_eps']).all()


def test_deformation_absorbed_by_private_nuisance():
    rng = np.random.default_rng(3)
    Js = [rng.normal(size=(4, 2)) for _ in range(3)]
    Bs = [np.zeros((4, 0)), rng.normal(size=(4, 1)), np.zeros((4, 0))]
    Gs = [np.zeros((4, 1)), 0.7 * Bs[1], np.zeros((4, 1))]   # deformation lives in sector 1's nuisance
    Ns = [spd(rng, 4) for _ in range(3)]
    assert abs(info_split(Js, Gs, Ns, Bs=Bs)['total'].item()) < 1e-12


def test_nu_one_systematic_is_non_identifiable():
    A = np.ones((2, 1)); C = np.diag([0.01, 0.09]) ** 1
    b = np.array([0.0, -1 / 6])
    assert np.isfinite(sigma_eps(A, C, b))
    assert sigma_eps(A, C, b, [np.array([0.0, 1.0])]) == np.inf
    assert sigma_eps(A, C, b, [np.array([1.0, 0.0])]) == np.inf


def test_p5_inflation_is_one_over_sin():
    rng = np.random.default_rng(4)
    A = np.ones((4, 1)); C = np.diag(rng.uniform(0.5, 2, 4))
    b = rng.normal(size=4); s = rng.normal(size=4)
    cphi = confusion(A, C, b, s)
    assert np.isclose(sigma_eps(A, C, b, [s]) / sigma_eps(A, C, b), 1 / np.sqrt(1 - cphi ** 2), rtol=1e-10)


# ---------------------------------------------------------------- scaling and invariance
def test_covariance_scaling():
    rng = np.random.default_rng(5)
    Js, Gs, Ns, Bs = random_problem(rng)
    s1 = info_split(Js, Gs, Ns, Bs=Bs)
    c = 3.7
    s2 = info_split(Js, Gs, [c * N for N in Ns], Bs=Bs)
    for key in ('total', 'within_abstract', 'between_abstract'):
        assert np.isclose(s2[key].item(), s1[key].item() / c, rtol=RTOL)
    assert s1['nu'] == s2['nu']
    sm1 = summary_from_sectors(Js, Gs, Ns, Bs=Bs); sm2 = summary_from_sectors(Js, Gs, [c * N for N in Ns], Bs=Bs)
    z = rng.normal(size=sm1['A'].shape[0])
    d1 = decompose_T2(z, sm1['A'], block_diag(*sm1['Sigmas']), sm1['b'])
    d2 = decompose_T2(z, sm2['A'], block_diag(*sm2['Sigmas']), sm2['b'])
    assert np.isclose(d2['T2'], d1['T2'] / c, rtol=1e-8) and np.isclose(d2['Tpar'], d1['Tpar'] / c, rtol=1e-8)


def test_reparametrisation_invariance():
    rng = np.random.default_rng(6)
    Js, Gs, Ns, Bs = random_problem(rng)
    Q = rng.normal(size=(2, 2)) + 2 * np.eye(2)               # theta -> Q theta
    Js2 = [J @ np.linalg.inv(Q) for J in Js]
    Bs2 = [B @ (rng.normal(size=(B.shape[1],) * 2) + 2 * np.eye(B.shape[1])) if B.shape[1] else B for B in Bs]
    s = 2.5                                                    # eps -> eps / s
    Gs2 = [s * G for G in Gs]
    a = info_split(Js, Gs, Ns, Bs=Bs); b_ = info_split(Js2, Gs2, Ns, Bs=Bs2)
    assert np.isclose(b_['between_abstract'].item(), s ** 2 * a['between_abstract'].item(), rtol=1e-8)
    assert np.isclose(b_['total'].item(), s ** 2 * a['total'].item(), rtol=1e-8)
    assert a['nu'] == b_['nu']


def test_summary_basis_invariance_of_statistics():
    rng = np.random.default_rng(7)
    Js = [rng.normal(size=(4, 2)) for _ in range(3)]; Gs = [rng.normal(size=(4, 1)) for _ in range(3)]
    Ns = [spd(rng, 4) for _ in range(3)]
    sm = summary_from_sectors(Js, Gs, Ns); C = block_diag(*sm['Sigmas'])
    M = block_diag(*[rng.normal(size=(2, 2)) + 2 * np.eye(2) for _ in range(3)])
    z = rng.normal(size=6); s = rng.normal(size=6)
    d1 = decompose_T2(z, sm['A'], C, sm['b']); d2 = decompose_T2(M @ z, M @ sm['A'], M @ C @ M.T, M @ sm['b'])
    for k in ('T2', 'Tpar', 'Tperp'):
        assert np.isclose(d1[k], d2[k], rtol=1e-9)
    assert np.isclose(confusion(sm['A'], C, sm['b'][:, 0], s), confusion(M @ sm['A'], M @ C @ M.T, M @ sm['b'][:, 0], M @ s), rtol=1e-9)


# ---------------------------------------------------------------- correlated residuals (P7)
def correlated_problem(rng, c=0.4, m=2):
    """Shared-noise model (always positive definite): n_k = n_k^indep + c T_k eta, eta ~ N(0, I_m).
    Each sector's own covariance is the diagonal block of the full one."""
    sizes = [3, 3, 4]
    Js = [rng.normal(size=(n, 2)) for n in sizes]; Gs = [rng.normal(size=(n, 1)) for n in sizes]
    D = block_diag(*[spd(rng, n) for n in sizes])
    T = rng.normal(size=(sum(sizes), m))
    Nf = D + c ** 2 * T @ T.T
    edges = np.cumsum([0] + sizes)
    Ns = [Nf[edges[i]:edges[i + 1], edges[i]:edges[i + 1]] for i in range(3)]
    return Js, Gs, Ns, Nf, sizes


def test_correlated_joint_fit_restores_split():
    rng = np.random.default_rng(8)
    Js, Gs, Ns, Nf, sizes = correlated_problem(rng)
    c0, c1, cb = asimov_split(Js, Gs, Ns, Nfull=Nf)
    W = np.linalg.inv(Nf); X1 = block_diag(*Js); F1 = X1.T @ W @ X1
    Lj = np.linalg.solve(F1, X1.T @ W)
    A = np.kron(np.ones((3, 1)), np.eye(2))
    Ib = asimov_summary_between(A, Lj @ np.vstack(Gs), np.linalg.inv(F1))
    assert np.isclose(Ib, cb, rtol=1e-8)


def test_correlated_size_correct_with_full_cov_and_wrong_without():
    rng = np.random.default_rng(9)
    Js, Gs, Ns, Nf, sizes = correlated_problem(rng, c=1.0)
    sm = summary_from_sectors(Js, Gs, Ns)
    Cf = summary_cov(sm['Ls'], Nf); Cb = block_diag(*sm['Sigmas'])
    nu = dof(sm['A'], Cf)
    Z = np.linalg.cholesky(Cf) @ rng.normal(size=(Cf.shape[0], 200_000))
    def size(C):
        R, Ci = residual_projector(sm['A'], C); r = R @ Z
        return np.mean(np.einsum('in,ij,jn->n', r, Ci, r) > stats.chi2.ppf(0.95, nu))
    assert abs(size(Cf) - 0.05) < 0.003
    assert abs(size(Cb) - 0.05) > 0.01


# ---------------------------------------------------------------- distribution laws (P4)
def test_cochran_split_with_profiled_template():
    rng = np.random.default_rng(10)
    A = np.ones((5, 1)); C = np.diag(rng.uniform(0.5, 2, 5)) ** 2
    b = rng.normal(size=(5, 1)); S = rng.normal(size=(5, 1))   # one competing template profiled
    nu = dof(A, C, S); assert nu == 3
    Z = np.linalg.cholesky(C) @ rng.normal(size=(5, 100_000)) + 0.8 * S   # template amplitude present
    R, Ci = residual_projector(A, C, S)
    U = R @ b; uq = U[:, 0] / np.sqrt(U[:, 0] @ Ci @ U[:, 0])
    r = R @ Z
    T2 = np.einsum('in,ij,jn->n', r, Ci, r); Tpar = (uq @ Ci @ r) ** 2; Tperp = T2 - Tpar
    assert abs(np.mean(Tpar > stats.chi2.ppf(0.95, 1)) - 0.05) < 0.004
    assert abs(np.mean(Tperp > stats.chi2.ppf(0.95, nu - 1)) - 0.05) < 0.004
    assert abs(np.corrcoef(Tpar, Tperp)[0, 1]) < 0.01
