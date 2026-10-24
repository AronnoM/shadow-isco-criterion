"""
Attack 3 -- statistical stress tests.
 (a) non-Gaussian EHT errors: actual size of nominal-5% tests built with a Gaussian approximation
 (b) template mismatch: power retained when the true deformation order differs from the assumed one
 (c) two deformation orders at once: correlation of (d2, d3) estimates
 (d) posterior-vs-likelihood double counting (EHT summary that already used the S2 prior)
 (e) Sgr A* in (ln M, ln D) form with GRAVITY and Keck covariances rebuilt from published errors
"""
import json
import numpy as np
from scipy import stats
from core import kernel, confusion, residual_projector, sigma_eps, decompose_T2

rng = np.random.default_rng(3)
out = {}
Nmc = 400_000

# ---------------------------------------------------------------- (a) non-Gaussian EHT errors
# sectors in theta_g (uas): GRAVITY, Keck-like (made consistent: same truth), flare, EHT
truth = 5.0
sig = np.array([0.0219, 0.0316, 2.0, 1.05])         # EHT symmetrised from +1.4/-0.7
A1 = np.ones((4, 1)); C1 = np.diag(sig ** 2)
u2 = np.array([kernel(2, 'orbit', 2800.0)] * 2 + [kernel(2, 'orbit', 9.0), kernel(2, 'shadow')]) * truth
R, Ci = residual_projector(A1, C1)
U = R @ u2
def stats_of(Z):
    r = R @ Z
    T2 = np.einsum('in,ij,jn->n', r, Ci, r)
    Tpar = (U @ Ci @ r) ** 2 / (U @ Ci @ U)
    return Tpar, T2 - Tpar
def eht_draw(model, n):
    if model == 'gaussian':
        return rng.normal(0, 1.05, n)
    if model == 'split-normal, mode at truth':
        up = rng.random(n) < 1.4 / 2.1
        return np.where(up, np.abs(rng.normal(0, 1.4, n)), -np.abs(rng.normal(0, 0.7, n)))
    if model == 'split-normal, median at truth':
        x = np.where(rng.random(n) < 1.4 / 2.1, np.abs(rng.normal(0, 1.4, n)), -np.abs(rng.normal(0, 0.7, n)))
        return x - np.median(x)
    if model == 'lognormal (sigma_ln = 0.21)':
        return truth * (np.exp(rng.normal(0, 0.21, n)) - 1)
print("(a) actual size of nominal 5% tests (Gaussian approximation used in the statistic)")
res_a = []
for model in ['gaussian', 'split-normal, mode at truth', 'split-normal, median at truth', 'lognormal (sigma_ln = 0.21)']:
    Z = np.vstack([rng.normal(0, sig[0], Nmc), rng.normal(0, sig[1], Nmc), rng.normal(0, sig[2], Nmc),
                   eht_draw(model, Nmc)]) + truth
    Tpar, Tperp = stats_of(Z)
    sz_par = float(np.mean(Tpar > stats.chi2.ppf(0.95, 1)))
    sz_perp = float(np.mean(Tperp > stats.chi2.ppf(0.95, 2)))
    res_a.append(dict(model=model, size_Tpar=sz_par, size_Tperp=sz_perp))
    print(f"   {model:32s}: size(T_par) = {sz_par:.3f}   size(T_perp) = {sz_perp:.3f}")
out['a_nongaussian'] = res_a

# ---------------------------------------------------------------- (b) template mismatch
print("\n(b) template mismatch: fraction of noncentrality kept by a d2-template test when truth is d_n")
designs = {
    'Sgr A* now (GRAVITY, Keck, flare 9M, EHT)': ([('orbit', 2800.0), ('orbit', 2800.0), ('orbit', 9.0), ('shadow',)],
                                                  [0.0043, 0.0064, 0.40, 0.20]),
    'ladder (S2, orbit 9M, ISCO-QPO, shadow)':   ([('orbit', 2800.0), ('orbit', 9.0), ('isco_freq',), ('shadow',)],
                                                  [0.005, 0.05, 0.05, 0.05]),
}
res_b = []
for name, (probes, sg) in designs.items():
    Ad = np.ones((len(probes), 1)); Cd = np.diag(np.array(sg) ** 2)
    def tmpl(n):
        return np.array([kernel(n, pr[0], *pr[1:]) for pr in probes])
    for n_true in [3, 4]:
        cos = confusion(Ad, Cd, tmpl(2), tmpl(n_true))
        res_b.append(dict(design=name, n_true=n_true, kept=cos ** 2))
        print(f"   {name}: truth n={n_true}: kept fraction cos^2 = {cos**2:.3f}")
out['b_mismatch'] = res_b

# ---------------------------------------------------------------- (c) two orders at once
print("\n(c) joint (d2, d3) estimate: correlation and error inflation")
res_c = []
for name, (probes, sg) in designs.items():
    Ad = np.ones((len(probes), 1)); Cd = np.diag(np.array(sg) ** 2)
    R, Ci = residual_projector(Ad, Cd)
    T = np.column_stack([[kernel(n, pr[0], *pr[1:]) for pr in probes] for n in (2, 3)])
    U = R @ T; F = U.T @ Ci @ U; Cov = np.linalg.inv(F)
    rho = Cov[0, 1] / np.sqrt(Cov[0, 0] * Cov[1, 1])
    s2_alone = 1 / np.sqrt(F[0, 0])
    res_c.append(dict(design=name, rho=rho, sigma_d2_alone=s2_alone, sigma_d2_joint=np.sqrt(Cov[0, 0]),
                      sigma_d3_joint=np.sqrt(Cov[1, 1]), nu=len(probes) - 1))
    print(f"   {name}: corr(d2,d3) = {rho:+.4f}; sigma(d2) {s2_alone:.3f} alone -> {np.sqrt(Cov[0,0]):.3f} joint;"
          f" dof left for systematics = {len(probes)-1-2}")
out['c_two_orders'] = res_c

# ---------------------------------------------------------------- (d) double counting
print("\n(d) EHT summary that already includes the GRAVITY prior, treated as independent")
sG, sE = 0.0219, 1.05
w = (1 / sG ** 2) / (1 / sG ** 2 + 1 / sE ** 2)
zG = rng.normal(0, sG, Nmc); zE = rng.normal(0, sE, Nmc)
zE_post = w * zG + (1 - w) * zE                      # 'posterior mean' of EHT using GRAVITY prior
sE_post = 1 / np.sqrt(1 / sG ** 2 + 1 / sE ** 2)
T_wrong = (zE_post - zG) ** 2 / (sE_post ** 2 + sG ** 2)
T_right = (zE - zG) ** 2 / (sE ** 2 + sG ** 2)
print(f"   prior weight w = {w:.4f}; mean T2 with double counting = {T_wrong.mean():.2e} (should be 1);"
      f" correct = {T_right.mean():.3f}")
# with a real offset of 1 sigma_E in EHT
off = sE
T_wrong_s = ((w * zG + (1 - w) * (zE + off)) - zG) ** 2 / (sE_post ** 2 + sG ** 2)
T_right_s = (zE + off - zG) ** 2 / (sE ** 2 + sG ** 2)
print(f"   power at a 1-sigma EHT offset: double-counted {np.mean(T_wrong_s > 3.841):.3f} vs correct {np.mean(T_right_s > 3.841):.3f}")
out['d_double_counting'] = dict(w=w, mean_T2_wrong=float(T_wrong.mean()), mean_T2_right=float(T_right.mean()),
                                power_wrong=float(np.mean(T_wrong_s > 3.841)), power_right=float(np.mean(T_right_s > 3.841)))

# ---------------------------------------------------------------- (e) Sgr A* in (ln M, ln D)
print("\n(e) Sgr A* with (ln M, ln D) as common parameters")
def cov_lnM_lnD(M, sM_stat, R0, sR_stat, sR_sys, theta_stat_frac, theta_sys_frac):
    """Rebuild cov(ln M, ln R0) from published numbers.
    stat part: correlation chosen so the stat error on ln theta_g = ln M - ln R0 matches the quoted one;
    sys part: fully correlated, M ∝ R0^k with k fixed by the quoted theta_g systematic."""
    a, b = sM_stat / M, sR_stat / R0
    rho = (a ** 2 + b ** 2 - theta_stat_frac ** 2) / (2 * a * b)
    S_stat = np.array([[a * a, rho * a * b], [rho * a * b, b * b]])
    k = 1 + theta_sys_frac / (sR_sys / R0)
    v = np.array([k * sR_sys / R0, sR_sys / R0])
    return S_stat + np.outer(v, v), rho, k
SG, rhoG, kG = cov_lnM_lnD(4.297, 0.012, 8277, 9, 33, 0.009 / 5.125, 0.020 / 5.125)
SK, rhoK, kK = cov_lnM_lnD(3.964, 0.047, 7946, 50, 32, 0.03 / 4.92, 0.01 / 4.92)
print(f"   GRAVITY: rho_stat = {rhoG:.3f}, sys along M ∝ R0^{kG:.2f};  Keck: rho_stat = {rhoK:.3f}, sys along M ∝ R0^{kK:.2f}")
print("   (GRAVITY M stat error 0.012e6 taken from the 0.25% quoted precision; check against the paper)")
G_, Mm = 6.674e-11, 1.989e30
uas = np.pi / 180 / 3600 / 1e6; pc = 3.0857e16; c_ = 2.998e8
const = np.log(G_ * Mm / c_ ** 2 / pc / uas)      # ln theta_g[uas] = ln M[Msun] - ln D[pc] + const
zG = np.array([np.log(4.297e6), np.log(8277.0)]); zK = np.array([np.log(3.964e6), np.log(7946.0)])
th_EHT, s_ln_EHT = 4.8, 0.21
z = np.concatenate([zG, zK, [np.log(th_EHT) - const]])
A = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, -1]], float)
C = np.zeros((5, 5)); C[:2, :2] = SG; C[2:4, 2:4] = SK; C[4, 4] = s_ln_EHT ** 2
b = np.array([kernel(2, 'orbit', 2800.0), 0, kernel(2, 'orbit', 2800.0), 0, kernel(2, 'shadow')])
dec = decompose_T2(z, A, C, b)
nu = A.shape[0] - np.linalg.matrix_rank(A)
D = zG - zK; Sd = SG + SK
chi2_GK = float(D @ np.linalg.solve(Sd, D)); e = np.array([1, -1])
chi2_GK_theta = float((e @ D) ** 2 / (e @ Sd @ e))
print(f"   T2 = {dec['T2']:.2f} (nu = {nu});  T2_par = {dec['Tpar']:.3f};  T2_perp = {dec['Tperp']:.2f}")
print(f"   GRAVITY vs Keck alone: 2-dof chi2 = {chi2_GK:.2f}, of which theta_g direction = {chi2_GK_theta:.2f}")
print(f"   d2 estimate = {dec['eps_hat'][0]:+.2f} +- {np.sqrt(dec['cov_eps'][0,0]):.2f}")
out['e_sgra_MD'] = dict(rho_stat_G=rhoG, k_G=kG, rho_stat_K=rhoK, k_K=kK, T2=dec['T2'], nu=int(nu),
                        Tpar=dec['Tpar'], Tperp=dec['Tperp'], chi2_GK=chi2_GK, chi2_GK_theta=chi2_GK_theta,
                        d2=float(dec['eps_hat'][0]), sigma_d2=float(np.sqrt(dec['cov_eps'][0, 0])))

json.dump(out, open('results/attack3.json', 'w'), indent=1, default=float)
