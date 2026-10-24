"""
Stellar-sector response to the n = 2 deformation (referee point 6A/6B), from the Keck S0-2 data.

 (1) inject:  the GR (delta_2 = 0) fit is applied to data generated with delta_2; the linear bias of the
     reported summaries is  b = d(ln M, ln R0)/d delta_2 = (J^T J)^-1 J^T g,  g = d(whitened model)/d delta_2.
 (2) within-sector information: delta_2 fitted together with all 14 orbit/frame parameters.
 (3) maximum-likelihood check of the noise parameters (Lambda, p) at the GR optimum.
Noise model and data selection as in s2_refit.py (Do et al. 2019).
"""
import json
import numpy as np
from scipy.optimize import least_squares, minimize
import s2_refit as S


def gr_fit(theta0):
    r = least_squares(S.residuals, np.array(theta0, float), x_scale='jac', method='lm', max_nfev=4000)
    for _ in range(4):
        r2 = least_squares(S.residuals, r.x, x_scale='jac', method='lm', max_nfev=4000)
        if float(r.fun @ r.fun) - float(r2.fun @ r2.fun) < 1e-6:
            r = r2; break
        r = r2
    return r


def analyse(theta0, label):
    r = gr_fit(theta0)
    th = r.x; chi2 = float(r.fun @ r.fun)
    err0 = np.sqrt(np.diag(np.linalg.pinv(r.jac.T @ r.jac)))
    J = S.jac_central(lambda p: S.residuals(p), th, 0.1 * err0)
    cov = np.linalg.inv(J.T @ J); err = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    h = 0.05
    g = (S.residuals(th, d2=h) - S.residuals(th, d2=-h)) / (2 * h)
    g2 = (S.residuals(th, d2=2 * h) - S.residuals(th, d2=-2 * h)) / (4 * h)
    # data y = f(theta0, delta); GR fit minimises |f(theta, 0) - y|^2 -> theta_hat - theta0 = (J^T J)^-1 J^T g delta
    dth = np.linalg.solve(J.T @ J, J.T @ g)
    b = np.array([dth[0] / th[0], dth[1] / th[1]])
    Jd = np.column_stack([J, g])
    sd, ratio = S.profiled_sigma_last(Jd)
    step = -np.linalg.lstsq(Jd, r.fun, rcond=None)[0]
    d_hat = float(step[-1])
    lin_err = float(np.max(np.abs(g - g2)) / np.max(np.abs(g)))
    out = dict(theta=th.tolist(), err=err.tolist(), chi2=chi2, dof=r.fun.size - len(th), rho=rho,
               b_lnM=b[0], b_lnR0=b[1], b_lnthetag=b[0] - b[1], sigma_d2=sd, d2_hat=d_hat, lin_check=lin_err,
               fsp_equiv=dict(f=1 - d_hat / 6, sigma=sd / 6))
    print(f"[{label}] chi2 = {chi2:.1f}/{out['dof']}; M = {th[0]:.3f}+-{err[0]:.3f}, R0 = {th[1]:.3f}+-{err[1]:.3f}, rho = {rho:.3f}")
    print(f"   GR-fit bias per unit delta_2: dlnM = {b[0]:+.5f}, dlnR0 = {b[1]:+.5f}, dln theta_g = {b[0]-b[1]:+.5f}"
          f"  (circular kernel at pericentre {-1/2900:+.5f})")
    print(f"   within-sector: delta_2 = {d_hat:+.2f} +- {sd:.2f}  (f_SP-equivalent {1-d_hat/6:.2f} +- {sd/6:.2f}); "
          f"derivative step check {lin_err:.1e}")
    return out, th


def neg2lnL(Lam, p, th):
    S.set_noise(Lam, p)
    X, Y, V = S.model(th)
    rX, rY = X - S.X_obs, Y - S.Y_obs
    lx = S.LXi @ rX; ly = S.LYi @ rY
    logdet = -2 * (np.sum(np.log(np.abs(np.diag(S.LXi)))) + np.sum(np.log(np.abs(np.diag(S.LYi)))))
    return float(lx @ lx + ly @ ly + logdet)


if __name__ == '__main__':
    v1 = json.load(open('results/s2_refit_v1.json'))
    th0 = v1['no_shell']['theta']
    out = {}
    S.set_noise(None)
    out['independent'], _ = analyse(th0, 'independent errors, 45/115')
    S.set_noise(28.0, 0.55)
    out['do_noise'], th = analyse(th0, 'Do et al. noise, Lambda = 28 mas, p = 0.55')
    # (3) ML noise parameters at the GR optimum (orbit held at the Do-noise optimum)
    f = lambda q: neg2lnL(np.exp(q[0]), 1 / (1 + np.exp(-q[1])), th)
    q = minimize(f, [np.log(28.0), np.log(0.55 / 0.45)], method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-4))
    Lml, pml = float(np.exp(q.x[0])), float(1 / (1 + np.exp(-q.x[1])))
    print(f"   ML noise at fixed orbit: Lambda = {Lml:.1f} mas, p = {pml:.2f}  (Do et al. medians 28, 0.55)")
    out['ml_noise'] = dict(Lambda=Lml, p=pml)
    S.set_noise(Lml, pml)
    out['ml_noise_fit'], _ = analyse(th, f'ML noise Lambda = {Lml:.0f}, p = {pml:.2f}')
    S.set_noise(28.0, 0.55)
    json.dump(out, open('results/s2_deform.json', 'w'), indent=1, default=float)
