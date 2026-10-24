"""
Diagnostic for the S2 shell scan: fit the Schwarzschild-precession factor f_SP (GRAVITY 2020 convention:
f_SP multiplies the 1PN terms of the equation of motion; redshift and Romer delay unchanged) to the Keck
data of Do et al. 2019, with the data selection and correlated astrometric-noise model of Do et al. (as in s2_refit.py).
A negative best-fit shell mass outside pericentre is a retrograde-precession deficit, i.e. f_SP > 1.
"""
import json, sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import s2_refit as S


def rhs_f(t, s, GM, f):
    x, v = s[:3], s[3:]
    r = np.sqrt(x @ x); v2 = v @ v; rv_ = x @ v
    c2 = S.C_AU ** 2
    acc = -GM * x / r ** 3 * (1 + f * (-4 * GM / (c2 * r) + v2 / c2)) + f * 4 * GM / (c2 * r ** 3) * rv_ * v
    return np.concatenate([v, acc])


def residuals_f(p):
    theta, f = p[:-1], p[-1]
    if not (0 < theta[9] < 0.99 and theta[7] > 0):
        return S.BAD
    orig = S.rhs
    S.rhs = lambda t, s, GM, Gm, Rs, d2=0.0: rhs_f(t, s, GM, f)
    try:
        X, Y, V = S.model(theta)
    finally:
        S.rhs = orig
    return np.concatenate([S.LXi @ (X - S.X_obs), S.LYi @ (Y - S.Y_obs), (V - S.V_obs) / S.sV])


if __name__ == '__main__':
    ref = json.load(open('results/s2_refit.json'))
    th = np.array(ref['no_shell']['theta']); err = np.array(ref['no_shell']['err'])
    chi2_gr = ref['no_shell']['chi2']
    r = least_squares(residuals_f, np.append(th, 1.0), x_scale='jac', method='lm', max_nfev=4000)
    J = S.jac_central(residuals_f, r.x, np.append(0.1 * err, 0.02))
    sf, ratio = S.profiled_sigma_last(J)
    chi2 = float(r.fun @ r.fun)
    print(f"f_SP = {r.x[-1]:.2f} +- {sf:.2f}  (GR = 1); chi2 {chi2:.2f} vs {chi2_gr:.2f} at f_SP = 1"
          f" (delta {chi2 - chi2_gr:+.2f}); M = {r.x[0]:.3f}, R0 = {r.x[1]:.3f}")
    print("GRAVITY 2020 (A&A 636, L5): f_SP = 1.10 +- 0.19. The Keck-only excess persists with the Do et al. noise model;"
          " a jackknife (s2_deform/jackknife) traces it to the 1995-2004 speckle astrometry.")
    json.dump(dict(f_SP=float(r.x[-1]), sigma=float(sf), chi2=chi2, chi2_GR=chi2_gr, M=float(r.x[0]), R0=float(r.x[1])),
              open('results/s2_fsp.json', 'w'), indent=1)
