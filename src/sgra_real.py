"""
Sgr A* consistency test with the published Keck posterior (Do et al. 2019, Data S3) replacing the
reconstructed covariance, and GRAVITY's M-R0 correlation digitised from GRAVITY 2022 Fig. B.1.

Sectors: GRAVITY (ln M, ln R0), Keck (ln M, ln R0), EHT image-only (ln theta_g). Common parameters (ln M, ln D).
Statistical covariances:
  Keck GR chain:  sd(lnM) = 0.011919, sd(lnR0) = 0.006271, cov = 7.264e-5 (rho = 0.972)   [exact, from chain]
  Keck Upsilon-free chain: sd 0.014699 / 0.007572, cov 1.0914e-4 (rho = 0.981)          [exact, from chain]
  GRAVITY 2022: sd(M) = 0.012e6, sd(R0) = 9 pc (stat); rho digitised 0.95-0.97 (central 0.96)
Systematics (not in any chain): Keck jackknife M 0.026e6, R0 32 pc; GRAVITY ~0.040e6 and ~30 pc
(GRAVITY 2022 conclusions) or theta_g sys 0.020 uas with R0 sys 33 pc (as quoted by EHT). Their
correlation structure is not published, so several models are scanned.
"""
import json, itertools
import numpy as np
from scipy import stats
from scipy.linalg import sqrtm
from core import kernel, decompose_T2

G_, Mm, c_, pc = 6.674e-11, 1.989e30, 2.998e8, 3.0857e16
uas = np.pi / 180 / 3600 / 1e6
CONST = np.log(G_ * Mm / c_ ** 2 / pc / uas)
chain = json.load(open('data/do2019_chain_summary.json'))
# stellar-sector response to delta_2 from injection into the Keck S0-2 model (s2_deform.py, Do et al. noise):
# d(ln M, ln R0)/d delta_2 of the GR fit; applied to both stellar sectors (both S2-dominated)
try:
    _dd = json.load(open('results/s2_deform.json'))['do_noise']
    B_STAR = np.array([_dd['b_lnM'], _dd['b_lnR0']])
except FileNotFoundError:
    B_STAR = np.array([-1 / 2900.0, 0.0])

KECK = {
    'GR chain': dict(mu=[chain['GR']['mean_lnM'] + np.log(1e6), chain['GR']['mean_lnR0'] + np.log(1e3)],
                     S=[[chain['GR']['sd_lnM'] ** 2, chain['GR']['cov_lnM_lnR0']],
                        [chain['GR']['cov_lnM_lnR0'], chain['GR']['sd_lnR0'] ** 2]],
                     M=chain['GR']['mean_M'], D=chain['GR']['mean_R0_kpc'] * 1e3),
    'Upsilon-free chain': dict(mu=[chain['Upsilon_free']['mean_lnM'] + np.log(1e6), chain['Upsilon_free']['mean_lnR0'] + np.log(1e3)],
                               S=[[chain['Upsilon_free']['sd_lnM'] ** 2, chain['Upsilon_free']['cov_lnM_lnR0']],
                                  [chain['Upsilon_free']['cov_lnM_lnR0'], chain['Upsilon_free']['sd_lnR0'] ** 2]],
                               M=chain['Upsilon_free']['mean_M'], D=chain['Upsilon_free']['mean_R0_kpc'] * 1e3),
}


def sys_cov(kind, M, D, sM, sD, th_sys_frac=None):
    """Systematic covariance in (ln M, ln D)."""
    a, b = sM / M, sD / D
    if kind == 'none':
        return np.zeros((2, 2))
    if kind == 'uncorrelated':
        return np.diag([a * a, b * b])
    if kind == 'full, M sys and D sys as quoted':
        v = np.array([a, b]); return np.outer(v, v)
    if kind == 'full, theta_g sys as quoted':
        k = 1 + th_sys_frac / b
        v = np.array([k * b, b]); return np.outer(v, v)
    raise ValueError(kind)


def sectors(keck='GR chain', rho_G=0.96, sysK='full, theta_g sys as quoted', sysG='full, theta_g sys as quoted'):
    """Means and covariances (stat + sys) of the GRAVITY and Keck sectors in (ln M[Msun], ln R0[pc])."""
    MG, DG = 4.297, 8277.0
    aG, bG = 0.012 / MG, 9.0 / DG
    SG = np.array([[aG * aG, rho_G * aG * bG], [rho_G * aG * bG, bG * bG]])
    thG = MG * 1e6 / DG * np.exp(CONST)
    if sysG == 'full, M sys and D sys as quoted':
        SG = SG + sys_cov(sysG, MG, DG, 0.040, 30.0)
    else:
        SG = SG + sys_cov(sysG, MG, DG, 0.040, 33.0, th_sys_frac=0.020 / thG)
    zG = np.log([MG * 1e6, DG])
    kk = KECK[keck]
    SK = np.array(kk['S'], float)
    thK = kk['M'] * 1e6 / kk['D'] * np.exp(CONST)
    SK = SK + sys_cov(sysK, kk['M'], kk['D'], 0.026, 32.0, th_sys_frac=0.01 / thK)
    zK = np.array(kk['mu'])
    return zG, SG, zK, SK, thG, thK


def analyse(keck='GR chain', rho_G=0.96, sysK='full, theta_g sys as quoted', sysG='full, theta_g sys as quoted',
            rho_GK=0.0, eht='lognormal', nuis='none', bstar_scale=1.0, bcal=0.0, K_GK=None):
    zG, SG, zK, SK, thG, thK = sectors(keck, rho_G, sysK, sysG)
    # EHT surrogate (idealised)
    th_obs = 4.8
    sE = {'lognormal': 0.21, 'split-normal upper side': np.log(6.2 / 4.8),
          'split-normal lower side': np.log(4.8 / 4.1), 'wide (x2)': 0.42}[eht]
    z = np.concatenate([zG, zK, [np.log(th_obs) - CONST]])
    A = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, -1]], float)
    C = np.zeros((5, 5)); C[:2, :2] = SG; C[2:4, 2:4] = SK; C[4, 4] = sE ** 2
    if rho_GK or K_GK is not None:
        Kx = rho_GK * np.eye(2) if K_GK is None else np.asarray(K_GK)
        X = np.real(sqrtm(SG)) @ Kx @ np.real(sqrtm(SK)); C[:2, 2:4] = X; C[2:4, :2] = X.T
    S = None
    if nuis == 'Keck distance-scale offset':
        S = np.array([0, 0, 2.0, 1.0, 0.0])
    elif nuis == 'EHT calibration 10%':
        C[4, 4] += 0.10 ** 2
    bs = bstar_scale * B_STAR
    b = np.array([bs[0], bs[1], bs[0], bs[1], kernel(2, 'shadow') + bcal])
    d = decompose_T2(z, A, C, b, S=S)
    p = stats.chi2.sf(d['T2'], d['nu'])
    # GRAVITY vs Keck alone, and the part along ln R0 at fixed M
    Dz = zG - zK; Sd = SG + SK
    if rho_GK or K_GK is not None:
        Sd = Sd - C[:2, 2:4] - C[2:4, :2]                     # C_diff = C_G + C_K - C_GK - C_GK^T
    chi2_GK = float(Dz @ np.linalg.solve(Sd, Dz))
    return dict(T2=d['T2'], nu=d['nu'], p=p, sigma=float(stats.norm.isf(p / 2)), Tpar=d['Tpar'], Tperp=d['Tperp'],
                d2=float(d['eps_hat'][0]), sd2=float(np.sqrt(d['cov_eps'][0, 0])), chi2_GK=chi2_GK,
                thG=thG, thK=thK)


if __name__ == '__main__':
    base = analyse()
    print("BASELINE (Keck GR chain + jackknife sys along the EHT-quoted theta_g sys; GRAVITY rho 0.96 + EHT-quoted sys;"
          " no shared-data correlation; log-normal EHT):")
    print(f"  theta_g GRAVITY {base['thG']:.3f}, Keck {base['thK']:.3f} uas")
    print(f"  T2 = {base['T2']:.2f} (nu = {base['nu']}), {base['sigma']:.1f} sigma; T_par = {base['Tpar']:.3f}; "
          f"T_perp = {base['Tperp']:.2f}; GRAVITY-vs-Keck chi2(2 dof) = {base['chi2_GK']:.2f}; d2 = {base['d2']:+.2f} +- {base['sd2']:.2f}")
    stat_only = analyse(sysK='none', sysG='none')
    print(f"  statistical errors only: T2 = {stat_only['T2']:.2f}, {stat_only['sigma']:.1f} sigma; T_par = {stat_only['Tpar']:.3f}")

    grid = dict(keck=list(KECK), rho_G=[0.95, 0.96, 0.97],
                sysK=['none', 'uncorrelated', 'full, M sys and D sys as quoted', 'full, theta_g sys as quoted'],
                sysG=['none', 'uncorrelated', 'full, M sys and D sys as quoted', 'full, theta_g sys as quoted'],
                rho_GK=[0.0, 0.2, 0.4], eht=['lognormal', 'split-normal upper side', 'split-normal lower side', 'wide (x2)'],
                nuis=['none', 'EHT calibration 10%', 'Keck distance-scale offset'])
    rows = []
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(grid.keys(), combo))
        r = analyse(**kw); r.update(kw); rows.append(r)
    print(f"\n{len(rows)} variants")
    for nuis in grid['nuis']:
        sel = [r for r in rows if r['nuis'] == nuis]
        sg = np.array([r['sigma'] for r in sel]); tp = np.array([r['Tpar'] for r in sel])
        d2 = np.array([r['d2'] for r in sel]); sd = np.array([r['sd2'] for r in sel])
        print(f"  {nuis:28s}: significance {sg.min():.1f}-{sg.max():.1f} sigma | T_par {tp.min():.3f}-{tp.max():.3f} |"
              f" d2 {d2.min():+.2f}..{d2.max():+.2f}, sigma(d2) {sd.min():.2f}-{sd.max():.2f}")
    sel = [r for r in rows if r['nuis'] == 'none']
    for factor in ['keck', 'rho_G', 'sysK', 'sysG', 'rho_GK', 'eht']:
        vals = {}
        for r in sel:
            vals.setdefault(str(r[factor]), []).append(r['sigma'])
        print(f"   by {factor:6s}: " + "; ".join(f"{k}: {min(v):.1f}-{max(v):.1f}" for k, v in vals.items()))
    # ---- manifest: every variant with its null hypothesis
    import csv
    with open('results/sgra_manifest.csv', 'w', newline='') as fh:
        keys = list(grid.keys()) + ['null_hypothesis', 'T2', 'nu', 'sigma', 'Tpar', 'd2', 'sd2']
        w = csv.writer(fh); w.writerow(keys)
        for r in rows:
            null = {'none': 'common (lnM, lnD)', 'EHT calibration 10%': 'common (lnM, lnD) + EHT calibration',
                    'Keck distance-scale offset': 'common (lnM, lnD) + free Keck offset along M ~ R0^2'}[r['nuis']]
            w.writerow([r[k] for k in grid.keys()] + [null] + [f"{r[k]:.6g}" for k in ['T2', 'nu', 'sigma', 'Tpar', 'd2', 'sd2']])
    summ = {}
    for nuis in grid['nuis']:
        sel_ = [r for r in rows if r['nuis'] == nuis]
        summ[nuis] = dict(n=len(sel_), sigma=[min(r['sigma'] for r in sel_), max(r['sigma'] for r in sel_)],
                          Tpar=[min(r['Tpar'] for r in sel_), max(r['Tpar'] for r in sel_)],
                          d2=[min(r['d2'] for r in sel_), max(r['d2'] for r in sel_)],
                          sd2=[min(r['sd2'] for r in sel_), max(r['sd2'] for r in sel_)])
    # ---- stellar response and EHT calibration response: bounding scans at baseline
    resp_scan = []
    for sc in [0.0, 1.0, 10.0, 30.0]:
        for bc in [-1 / 12, 0.0, 1 / 12]:
            r = analyse(bstar_scale=sc, bcal=bc)
            resp_scan.append(dict(bstar_scale=sc, bcal=bc, Tpar=r['Tpar'], d2=r['d2'], sd2=r['sd2'], sigma=r['sigma']))
            print(f"   stellar response x{sc:4.0f}, EHT calibration response {bc:+.3f}: T_par = {r['Tpar']:.4f},"
                  f" d2 = {r['d2']:+.2f} +- {r['sd2']:.2f}")
    # ---- general cross-covariance between the stellar groups: X = S_G^1/2 K S_K^1/2, ||K||_2 <= 1
    from scipy.optimize import minimize
    def K_of(p, psd):
        M = np.array(p).reshape(2, 2)
        if psd:
            M = M @ M.T
        s = np.linalg.norm(M, 2)
        return M / max(1.0, s / 0.999)
    def t2(p, psd):
        return analyse(K_GK=K_of(p, psd))['T2']
    best = {}
    rng = np.random.default_rng(1)
    for psd in (True, False):
        vals = []
        for _ in range(40):
            p0 = rng.normal(size=4)
            rr = minimize(t2, p0, args=(psd,), method='Nelder-Mead', options=dict(maxiter=600, xatol=1e-6, fatol=1e-6))
            vals.append((rr.fun, rr.x))
        f, x = min(vals, key=lambda v: v[0])
        r = analyse(K_GK=K_of(x, psd))
        best['psd' if psd else 'any'] = dict(T2=r['T2'], sigma=r['sigma'], K=K_of(x, psd).tolist())
        print(f"   cross-covariance {'K positive semidefinite' if psd else 'any contraction K':25s}: min T2 = {r['T2']:.2f}"
              f" ({r['sigma']:.1f} sigma), K = {np.round(K_of(x, psd), 3).tolist()}")
    worst = min([r for r in rows if r['nuis'] == 'none'], key=lambda r: r['sigma'])
    kw = {k: worst[k] for k in ['keck', 'rho_G', 'sysK', 'sysG', 'eht']}
    f = lambda p: analyse(K_GK=K_of(p, False), **kw)['T2']
    rng = np.random.default_rng(2)
    rr = min((minimize(f, rng.normal(size=4), method='Nelder-Mead', options=dict(maxiter=800, xatol=1e-7, fatol=1e-7))
              for _ in range(60)), key=lambda r: r.fun)
    r = analyse(K_GK=K_of(rr.x, False), **kw)
    best['any_least_significant_variant'] = dict(variant=kw, T2=r['T2'], sigma=r['sigma'], Tpar=r['Tpar'])
    print(f"   cross-covariance, any K, least significant configuration: T2 = {r['T2']:.2f} ({r['sigma']:.2f} sigma)")
    json.dump(dict(base=base, stat_only=stat_only, rows=rows, summary=summ, response_scan=resp_scan,
                   cross_cov_min=best, B_STAR=B_STAR.tolist()), open('results/sgra_real.json', 'w'), indent=1, default=float)
