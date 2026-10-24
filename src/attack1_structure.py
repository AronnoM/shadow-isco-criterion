"""
Attack 1 -- structural tests of the information split.
 (a) rank-deficient sectors (a sector that measures only some combinations of theta)
 (b) noise correlated across sectors
 (c) re-partitioning one sector into two
 (d) private nuisance (systematic) in one sector  <->  confusion angle
"""
import json
import numpy as np
from core import (info_split, summary_from_sectors, between_info_summary, summary_cov,
                  block_diag, residual_projector, confusion, sigma_eps, rank)

rng = np.random.default_rng(7)
out = {}


def spd(n, cond=5.0):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    ev = np.exp(rng.uniform(0, np.log(cond), n))
    return Q @ np.diag(ev) @ Q.T


def make_sector(n_obs, p, r, q=1):
    """observables n_obs, theta dim p, Jacobian rank r."""
    J = rng.normal(size=(n_obs, r)) @ rng.normal(size=(r, p))
    G = rng.normal(size=(n_obs, q))
    N = spd(n_obs)
    return J, G, N


# ---------------- (a) rank-deficient sectors
p = 2
specs = [(3, 2), (1, 1), (4, 2), (2, 1)]     # (n_obs, rank)
worst = 0.0; worst_sum = 0.0
for trial in range(200):
    secs = [make_sector(n, p, r) for n, r in specs]
    Js, Gs, Ns = map(list, zip(*secs))
    # skip draws where a nominally full-rank sector is numerically near-degenerate
    svr = [np.linalg.svd(J.T @ np.linalg.inv(N) @ J, compute_uv=False) for J, _, N in secs]
    if any(sv[rr - 1] / sv[0] < 1e-6 for sv, (_, rr) in zip(svr, specs)):
        continue
    s = info_split(Js, Gs, Ns)
    summ = summary_from_sectors(Js, Gs, Ns)
    C = block_diag(*summ['Sigmas'])
    Ib = between_info_summary(summ['A'], summ['b'], C).item()
    tot, wit = s['total'].item(), s['within'].item()
    worst = max(worst, abs(Ib - s['between_abstract'].item()) / tot)
    worst_sum = max(worst_sum, abs(wit + Ib - tot) / tot)
nu_formula = sum(r for _, r in specs) - p
out['a_rank_deficient'] = dict(
    sectors=specs, nu=s['nu'], nu_formula_sum_r_minus_rankA=nu_formula,
    max_rel_err_summary_vs_observable_between=worst, max_rel_err_within_plus_between_vs_total=worst_sum)
print("(a) rank-deficient sectors", specs)
print(f"    nu (observable space) = {s['nu']},  sum r_k - rank(A) = {nu_formula}")
print(f"    max rel. error | between(summary) - between(observable) | / total = {worst:.1e}")
print(f"    max rel. error | within + between - total | / total            = {worst_sum:.1e}")

# ---------------- (b) noise correlated across sectors
print("\n(b) cross-sector noise correlation")
res_b = []
specs_b = [(3, 2), (3, 2), (2, 1)]
for c in [0.0, 0.1, 0.3, 0.5, 0.7]:
    ratios, deficits = [], []
    for trial in range(300):
        secs = [make_sector(n, p, r) for n, r in specs_b]
        Js, Gs, Ns = map(list, zip(*secs))
        sizes = [N.shape[0] for N in Ns]
        # build a full covariance with given cross-correlation strength c
        D = block_diag(*Ns)
        L = np.linalg.cholesky(D)
        n = D.shape[0]
        Z = rng.normal(size=(n, n)); X = Z @ Z.T / n
        # cross blocks from X scaled to keep PD: N_full = L (I + c*Offdiag(X_normalised)) L^T
        Xs = X / np.sqrt(np.outer(np.diag(X), np.diag(X)))
        mask = 1 - block_diag(*[np.ones((k, k)) for k in sizes])
        Mmat = np.eye(n) + c * Xs * mask
        ev = np.linalg.eigvalsh(Mmat)
        if ev.min() <= 0.05:
            continue
        Nfull = L @ Mmat @ L.T
        s = info_split(Js, Gs, Ns, Nfull=Nfull)
        summ = summary_from_sectors(Js, Gs, Ns)
        Cfull = summary_cov(summ['Ls'], Nfull, sizes)
        Ib_correct = between_info_summary(summ['A'], summ['b'], Cfull).item()
        tot = s['total'].item(); wit = s['within'].item()
        ratios.append((wit + Ib_correct) / tot)
        # what happens if correlations are IGNORED in the summary test
        Cblk = block_diag(*summ['Sigmas'])
        R, Ci = residual_projector(summ['A'], Cblk)
        deficits.append(Ib_correct)
    ratios = np.array(ratios)
    res_b.append(dict(c=c, n=len(ratios), ratio_median=float(np.median(ratios)),
                      ratio_min=float(ratios.min()), ratio_max=float(ratios.max()),
                      frac_ratio_gt_1=float(np.mean(ratios > 1 + 1e-9))))
    print(f"    c={c:.1f}: (operational within + summary between)/total  median={np.median(ratios):.3f}"
          f"  range=[{ratios.min():.3f}, {ratios.max():.3f}]  frac>1: {np.mean(ratios > 1+1e-9):.2f}")
out['b_correlated'] = res_b

# size of the summary test when cross-correlations are ignored
print("    size of nominal-5% omnibus test when cross-correlations are ignored:")
from scipy import stats
size_res = []
for c in [0.0, 0.5, 1.0, 1.5]:
    # shared-noise model (always positive definite): n = n_indep + c T eta
    secs = [make_sector(n, p, r) for n, r in specs_b]
    Js, Gs, Ns0 = map(list, zip(*secs))
    sizes = [N.shape[0] for N in Ns0]
    T = rng.normal(size=(sum(sizes), 2))
    Nfull = block_diag(*Ns0) + c ** 2 * T @ T.T
    edges = np.cumsum([0] + sizes)
    Ns = [Nfull[edges[i]:edges[i + 1], edges[i]:edges[i + 1]] for i in range(len(sizes))]
    summ = summary_from_sectors(Js, Gs, Ns)
    Cfull = summary_cov(summ['Ls'], Nfull, sizes); Cblk = block_diag(*summ['Sigmas'])
    R, Ci = residual_projector(summ['A'], Cblk)
    nu = summ['A'].shape[0] - rank(summ['A'])
    zs = np.linalg.cholesky(Cfull) @ rng.normal(size=(Cfull.shape[0], 100000))
    rr = R @ zs
    T2 = np.einsum('in,ij,jn->n', rr, Ci, rr)
    sz = float(np.mean(T2 > stats.chi2.ppf(0.95, nu)))
    # strength of the induced cross-sector correlation (largest canonical correlation, sectors 0-1)
    size_res.append(dict(c=c, size=sz))
    print(f"      shared-noise amplitude c={c:.1f}: actual size = {sz:.3f}")
# range over random shared-noise loadings (sign and structure of cross-correlation vary)
sizes_rand = []
for trial in range(30):
    secs = [make_sector(n, p, r) for n, r in specs_b]
    Js, Gs, Ns0 = map(list, zip(*secs)); sizes = [N.shape[0] for N in Ns0]
    T = rng.normal(size=(sum(sizes), 2)); Nfull = block_diag(*Ns0) + T @ T.T
    edges = np.cumsum([0] + sizes)
    Ns = [Nfull[edges[i]:edges[i + 1], edges[i]:edges[i + 1]] for i in range(len(sizes))]
    summ = summary_from_sectors(Js, Gs, Ns)
    Cfull = summary_cov(summ['Ls'], Nfull, sizes); Cblk = block_diag(*summ['Sigmas'])
    R, Ci = residual_projector(summ['A'], Cblk); nu = summ['A'].shape[0] - rank(summ['A'])
    zs = np.linalg.cholesky(Cfull) @ rng.normal(size=(Cfull.shape[0], 40000)); rr = R @ zs
    sizes_rand.append(float(np.mean(np.einsum('in,ij,jn->n', rr, Ci, rr) > stats.chi2.ppf(0.95, nu))))
print(f"      30 random shared-noise structures (c=1): size ranges {min(sizes_rand):.3f} to {max(sizes_rand):.3f}")
size_res.append(dict(random_structures_min=min(sizes_rand), random_structures_max=max(sizes_rand)))
out['b_size_if_ignored'] = size_res

# ---------------- (c) re-partitioning
print("\n(c) splitting one sector into two")
res_c = []
for trial in range(5):
    J6, G6, N6 = make_sector(6, p, 2)
    N6 = block_diag(N6[:3, :3], N6[3:, 3:])          # independent halves so the split is legal
    other = [make_sector(3, p, 2), make_sector(2, p, 1)]
    Js1 = [J6] + [o[0] for o in other]; Gs1 = [G6] + [o[1] for o in other]; Ns1 = [N6] + [o[2] for o in other]
    Js2 = [J6[:3], J6[3:]] + Js1[1:]; Gs2 = [G6[:3], G6[3:]] + Gs1[1:]
    Ns2 = [N6[:3, :3], N6[3:, 3:]] + Ns1[1:]
    s1 = info_split(Js1, Gs1, Ns1); s2 = info_split(Js2, Gs2, Ns2)
    row = dict(total_before=s1['total'].item(), total_after=s2['total'].item(),
               within_before=s1['within'].item(), within_after=s2['within'].item(),
               between_before=s1['between_abstract'].item(), between_after=s2['between_abstract'].item(),
               nu_before=s1['nu'], nu_after=s2['nu'])
    res_c.append(row)
    print(f"    total {row['total_before']:.4f} -> {row['total_after']:.4f} | within {row['within_before']:.4f}"
          f" -> {row['within_after']:.4f} | between {row['between_before']:.4f} -> {row['between_after']:.4f}"
          f" | nu {row['nu_before']} -> {row['nu_after']}")
out['c_partition'] = res_c

# ---------------- (d) private nuisance in one sector == confusion angle
print("\n(d) marginalising a sector-j systematic: sigma(eps) inflates by 1/sin(phi)")
res_d = []
for trial in range(5):
    secs = [make_sector(n, p, r) for n, r in [(3, 2), (3, 2), (1, 1)]]
    Js, Gs, Ns = map(list, zip(*secs))
    summ = summary_from_sectors(Js, Gs, Ns)
    A, b = summ['A'], summ['b'][:, 0]; C = block_diag(*summ['Sigmas'])
    j0 = 4                                   # summary index of the rank-1 sector (offset systematic)
    e = np.zeros(A.shape[0]); e[j0] = 1.0
    cosphi = confusion(A, C, b, e)
    s0 = sigma_eps(A, C, b); s1 = sigma_eps(A, C, b, [e])
    # observable-space version: private nuisance B_3 = J_3-direction offset in sector 3
    Bs = [np.zeros((3, 0)), np.zeros((3, 0)), Js[2] @ np.linalg.pinv(summ['Ls'][2] @ Js[2]) @ np.ones((1, 1))]
    so = info_split(Js, Gs, Ns, Bs=Bs)
    sb_obs = 1/np.sqrt(so['between_abstract'].item())
    res_d.append(dict(cos_phi=cosphi, ratio_sigma=s1/s0, one_over_sin=1/np.sqrt(1-cosphi**2),
                      sigma_between_summary_marg=s1, sigma_between_observable_marg=sb_obs))
    print(f"    |cos phi|={cosphi:.3f}: sigma ratio={s1/s0:.4f}, 1/sin(phi)={1/np.sqrt(1-cosphi**2):.4f};"
          f" summary vs observable-space marginalised sigma: {s1:.4f} vs {sb_obs:.4f}")
out['d_nuisance'] = res_d

json.dump(out, open('results/attack1.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
