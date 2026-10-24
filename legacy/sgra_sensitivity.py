"""
Stage 3: Sgr A* significance under every input version, covariance assumption, EHT likelihood
and nuisance model. Common parameters (ln M, ln D); sectors GRAVITY (M, D), Keck (M, D), EHT (theta_g).

Inputs (sources in the notes):
  GRAVITY 2022 as quoted by EHT Sgr A* Paper I: M = 4.297 +- 0.013 e6, D = 8277 +- 9 +- 33 pc
  Keck variants (Do et al. 2019): Upsilon-free max-likelihood 3.984/7971, Upsilon-free median 3.975/7959,
      GR-assumed (text) 3.964/7946, EHT-quoted 3.951/7935
  EHT image-only theta_g = 4.8 (+1.4/-0.7) uas  -- an idealised surrogate likelihood (see notes)
"""
import json, itertools
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from core import kernel, decompose_T2, residual_projector

G_, Mm, c_, pc = 6.674e-11, 1.989e30, 2.998e8, 3.0857e16
uas = np.pi / 180 / 3600 / 1e6
CONST = np.log(G_ * Mm / c_ ** 2 / pc / uas)          # ln theta_g = ln M - ln D + CONST

GRAV = dict(M=4.297, sMs=0.013, D=8277, sDs=9, sDsys=33, th_stat=0.009, th_sys=0.020)
KECK = {
    'Ups-free ML':     dict(M=3.984, sMs=0.058, D=7971, sDs=59, sDsys=32, th_stat=None, th_sys=None, sMsys=0.026),
    'Ups-free median': dict(M=3.975, sMs=0.058, D=7959, sDs=59, sDsys=32, th_stat=None, th_sys=None, sMsys=0.026),
    'GR (text)':       dict(M=3.964, sMs=0.047, D=7946, sDs=50, sDsys=32, th_stat=None, th_sys=None, sMsys=0.026),
    'EHT-quoted':      dict(M=3.951, sMs=0.047, D=7935, sDs=50, sDsys=32, th_stat=0.03, th_sys=0.01, sMsys=0.026),
}


def cov(inp, rho_stat=None, k_sys=None):
    """cov(ln M, ln D). Statistical correlation either given or chosen to reproduce a quoted stat theta_g error
    (falls back to the GRAVITY-derived 0.97 when no theta_g error is quoted); systematic fully correlated
    along M ∝ D^k (k given, or fixed by a quoted theta_g systematic, or from the quoted M and D systematics)."""
    a, b = inp['sMs'] / inp['M'], inp['sDs'] / inp['D']
    th = inp['M'] * 1e6 / inp['D'] * np.exp(CONST)
    if rho_stat is None:
        rho_stat = (a * a + b * b - (inp['th_stat'] / th) ** 2) / (2 * a * b) if inp['th_stat'] else 0.97
        rho_stat = float(np.clip(rho_stat, -0.999, 0.999))
    sD = inp['sDsys'] / inp['D']
    if k_sys is None:
        if inp['th_sys']:
            k_sys = 1 + (inp['th_sys'] / th) / sD
        elif inp.get('sMsys'):
            k_sys = (inp['sMsys'] / inp['M']) / sD
        else:
            k_sys = 2.0
    S = np.array([[a * a, rho_stat * a * b], [rho_stat * a * b, b * b]])
    v = np.array([k_sys * sD, sD])
    return S + np.outer(v, v), rho_stat, k_sys


def analyse(keck, rho_stat=None, k_sys=None, rho_GK=0.0, eht='lognormal', nuis='none', sMs_G=0.013):
    SG, rG, kG = cov(dict(GRAV, sMs=sMs_G), rho_stat, k_sys)
    SK, rK, kK = cov(KECK[keck], rho_stat, k_sys)
    zG = np.log([GRAV['M'] * 1e6, GRAV['D']]); zK = np.log([KECK[keck]['M'] * 1e6, KECK[keck]['D']])
    # EHT surrogate likelihood in ln theta_g
    th_obs = 4.8
    if eht == 'lognormal':
        zE, sE = np.log(th_obs), 0.21
    elif eht == 'split-normal (upper side)':        # common fit lies above 4.8 -> the +1.4 side applies
        zE, sE = np.log(th_obs), np.log((th_obs + 1.4) / th_obs)
    elif eht == 'split-normal (lower side)':
        zE, sE = np.log(th_obs), np.log(th_obs / (th_obs - 0.7))
    elif eht == 'wide (x2)':
        zE, sE = np.log(th_obs), 0.42
    z = np.concatenate([zG, zK, [zE - CONST]])
    A = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, -1]], float)
    C = np.zeros((5, 5)); C[:2, :2] = SG; C[2:4, 2:4] = SK; C[4, 4] = sE ** 2
    # cross-group correlation from shared data (applied to matching components)
    if rho_GK:                                      # shared-data correlation, PD for |rho_GK| < 1
        from scipy.linalg import sqrtm
        X = rho_GK * np.real(sqrtm(SG)) @ np.real(sqrtm(SK))
        C[:2, 2:4] = X; C[2:4, :2] = X.T
    S = None
    if nuis == 'group distance-scale offset':      # free relative offset of Keck's ln D (and ln M ∝ 2 ln D)
        S = np.array([0, 0, 2.0, 1.0, 0.0])
    elif nuis == 'EHT calibration 10%':
        C[4, 4] += 0.10 ** 2
    b = np.array([kernel(2, 'orbit', 2800.0), 0, kernel(2, 'orbit', 2800.0), 0, kernel(2, 'shadow')])
    d = decompose_T2(z, A, C, b, S=S)
    p = stats.chi2.sf(d['T2'], d['nu'])
    return dict(T2=d['T2'], nu=d['nu'], p=p, sigma=stats.norm.isf(p / 2), Tpar=d['Tpar'], Tperp=d['Tperp'],
                d2=float(d['eps_hat'][0]), sd2=float(np.sqrt(d['cov_eps'][0, 0])), rhoG=rG, kG=kG, rhoK=rK, kK=kK)


rows = []
for keck, rho, k, rgk, eht, nuis, smg in itertools.product(
        list(KECK), [None, 0.90, 0.99], [None, 1.5, 2.5], [0.0, 0.2, 0.4],
        ['lognormal', 'split-normal (upper side)', 'split-normal (lower side)', 'wide (x2)'],
        ['none', 'EHT calibration 10%', 'group distance-scale offset'], [0.011, 0.013]):
    r = analyse(keck, rho, k, rgk, eht, nuis, smg)
    r.update(keck=keck, rho_stat=rho, k_sys=k, rho_GK=rgk, eht=eht, nuis=nuis, sMs_G=smg)
    rows.append(r)

base = analyse('GR (text)')
print(f"baseline (Keck GR text, derived covariances, log-normal EHT, no nuisance): T2 = {base['T2']:.2f}, nu = {base['nu']},"
      f" {base['sigma']:.1f} sigma; T_par = {base['Tpar']:.3f}; d2 = {base['d2']:+.2f} +- {base['sd2']:.2f}")
print(f"   derived: GRAVITY rho_stat {base['rhoG']:.3f}, k {base['kG']:.2f}; Keck rho_stat {base['rhoK']:.3f}, k {base['kK']:.2f}")
for nuis in ['none', 'EHT calibration 10%', 'group distance-scale offset']:
    sel = [r for r in rows if r['nuis'] == nuis]
    T2 = np.array([r['T2'] for r in sel]); sg = np.array([r['sigma'] for r in sel])
    Tp = np.array([r['Tpar'] for r in sel]); d2 = np.array([r['d2'] for r in sel]); sd = np.array([r['sd2'] for r in sel])
    print(f"{nuis:30s}: {len(sel)} variants | significance {sg.min():.1f}-{sg.max():.1f} sigma | T_par {Tp.min():.3f}-{Tp.max():.3f}"
          f" | d2 {d2.min():+.2f} to {d2.max():+.2f}, sigma(d2) {sd.min():.2f}-{sd.max():.2f}")
# which factor matters most for the significance (no nuisance)
sel = [r for r in rows if r['nuis'] == 'none']
for factor in ['keck', 'rho_stat', 'k_sys', 'rho_GK', 'eht', 'sMs_G']:
    vals = {}
    for r in sel:
        vals.setdefault(str(r[factor]), []).append(r['sigma'])
    print(f"  by {factor:8s}: " + "; ".join(f"{kk}: {np.min(v):.1f}-{np.max(v):.1f}" for kk, v in vals.items()))
json.dump(dict(base=base, rows=rows), open('results/sgra_sensitivity.json', 'w'), indent=1, default=float)
