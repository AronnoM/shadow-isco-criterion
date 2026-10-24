"""
S0-2 orbit refit to the Keck data released with Do et al. 2019 (Data S1: RVs, Data S2: astrometry),
with an optional static thin shell of mass m (fraction of the black-hole mass) at radius R_s.

Model (follows the Do et al. 2019 supplement except where noted):
  * 1PN test-particle equations of motion from Schwarzschild in harmonic coordinates (as Do et al.);
    shell: extra force -G m x/r^3 for r > R_s, potential G m / max(r, R_s)
  * Romer delay (iterated; Do et al. use one iteration, < 20 min difference); Shapiro delay neglected
  * relativistic redshift (v^2/2 + Phi)/c with Upsilon = 1 (the GR-fixed fit)
  * frame offsets x0, y0, vx0, vy0 (t_ref = 2000.0), vz0, and a NIRC2 RV offset (as Do et al.)
  * RV errors: Data S1 errors plus 11 km/s in quadrature for OSIRIS and NIFS (as Do et al.)
  * data selection as Do et al.: 45 astrometric epochs (the released table also lists 2018.673, which they
    exclude) and the 115 RVs of 2000-2018
  * astrometric noise: Do et al. Eq. S42, rho_ij = (1-p) delta_ij + p exp(-d_ij/Lambda), d_ij the projected
    separation of the measured positions, Sigma_ij = sigma_i sigma_j rho_ij for each axis; set_noise(Lam, p)
    (Lam = None gives independent errors). Default: Do et al. medians Lambda = 28 mas, p = 0.55
  * optional n = 2 deformation A = 1 - 2M/r + d2 (M/r)^2: PPN beta = 1 + d2/2, gamma = 1, i.e. an extra
    1PN acceleration + d2 (GM)^2 x / (c^2 r^4); its O(M^2/r^2) redshift change is neglected (2PN)
  * P is the osculating Keplerian period at the pericentre epoch T0, not the anomalistic period
Units: AU, yr, Msun; G Msun = 4 pi^2.
"""
import json, sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

C_AU = 63241.077            # AU / yr
KMS = 1 / 4.740470463       # (km/s) in AU/yr
GM1 = 4 * np.pi ** 2
TREF = 2000.0

rv = np.genfromtxt('data/do2019_rv.csv', delimiter=',', names=True, dtype=None, encoding=None)
ast = np.genfromtxt('data/do2019_astrometry.csv', delimiter=',', names=True, dtype=None, encoding=None)
mjd2yr = lambda mjd: 2000.0 + (mjd - 51544.5) / 365.25
ast = ast[np.abs(mjd2yr(ast['mjd']) - 2018.673) > 0.005]        # excluded by Do et al. (S0-2 anomalously bright)
rv = rv[mjd2yr(rv['mjd']) < 2019.0]                              # Do et al. use 2000-2018
t_ast = mjd2yr(ast['mjd']); X_obs = ast['x'] * 1e3; Y_obs = ast['y'] * 1e3       # mas
sX = ast['xerr'] * 1e3; sY = ast['yerr'] * 1e3
t_rv = mjd2yr(rv['mjd']); V_obs = rv['vlsr'].astype(float)
src_rv = np.array([s.strip() for s in rv['source']])
is_nirc2 = src_rv == 'NIRC2'
# Do et al. 2019 supplement: additive 11 km/s for OSIRIS and NIFS (sky/gas residual simulations)
add = np.where(np.char.startswith(src_rv, 'OSIRIS') | (src_rv == 'NIFS'), 11.0, 0.0)
sV = np.sqrt(rv['vzerr'].astype(float) ** 2 + add ** 2)

def set_noise(Lam=28.0, p=0.55):
    """Whitening matrices for the astrometric covariance of Do et al. (Eq. S42)."""
    global LXi, LYi, NOISE
    NOISE = (Lam, p)
    if Lam is None:
        LXi, LYi = np.diag(1 / sX), np.diag(1 / sY); return
    d = np.hypot(X_obs[:, None] - X_obs[None, :], Y_obs[:, None] - Y_obs[None, :])
    rho = (1 - p) * np.eye(len(X_obs)) + p * np.exp(-d / Lam)
    LXi = np.linalg.inv(np.linalg.cholesky(np.outer(sX, sX) * rho))
    LYi = np.linalg.inv(np.linalg.cholesky(np.outer(sY, sY) * rho))


set_noise()

NAMES = ['M6', 'R0', 'x0', 'y0', 'vx0', 'vy0', 'vz0', 'P', 'T0', 'e', 'inc', 'om', 'Om', 'dNIRC2']


def initial_state(M_in, P, T0, e, inc, om, Om):
    """Pericentre state (sky frame: X = RA, Y = Dec, Z = away from observer) for a Keplerian orbit."""
    a = (M_in * P ** 2) ** (1 / 3)
    rp = a * (1 - e); vp = np.sqrt(GM1 * M_in * (1 + e) / rp)
    i, w, W = np.radians([inc, om, Om])
    # unit vectors of the orbital frame in sky coordinates (Thiele-Innes, Dec = 'x', RA = 'y' convention)
    P_vec = np.array([np.sin(W) * np.cos(w) + np.cos(W) * np.sin(w) * np.cos(i),     # RA
                      np.cos(W) * np.cos(w) - np.sin(W) * np.sin(w) * np.cos(i),     # Dec
                      np.sin(w) * np.sin(i)])                                         # Z
    Q_vec = np.array([-np.sin(W) * np.sin(w) + np.cos(W) * np.cos(w) * np.cos(i),
                      -np.cos(W) * np.sin(w) - np.sin(W) * np.cos(w) * np.cos(i),
                      np.cos(w) * np.sin(i)])
    return np.concatenate([rp * P_vec, vp * Q_vec])


def rhs(t, s, GM, Gm, Rs, d2=0.0):
    """1PN Schwarzschild (harmonic) acceleration with the enclosed mass: outside the shell the exterior
    metric is Schwarzschild with M + m (Birkhoff), so the PN terms use the total mass there; inside the
    shell the potential is constant (its clock-rate effect on the dynamics is O(m/R * v^2/c^2), neglected)."""
    x, v = s[:3], s[3:]
    r = np.sqrt(x @ x); v2 = v @ v; rv_ = x @ v
    GMe = GM + Gm if r > Rs else GM
    acc = -GMe * x / r ** 3 * (1 - 4 * GMe / (C_AU ** 2 * r) + v2 / C_AU ** 2) + 4 * GMe / (C_AU ** 2 * r ** 3) * rv_ * v
    if d2:
        acc = acc + d2 * GMe ** 2 / (C_AU ** 2 * r ** 4) * x
    return np.concatenate([v, acc])


def model(theta, m_frac=0.0, Rs=np.inf, t_ast_=t_ast, t_rv_=t_rv, d2=0.0):
    M6, R0, x0, y0, vx0, vy0, vz0, P, T0, e, inc, om, Om, dN = theta
    GM = GM1 * M6 * 1e6; Gm = GM * m_frac
    # osculating Kepler initialisation at pericentre uses the mass enclosed there
    rp0 = (M6 * 1e6 * P ** 2) ** (1 / 3) * (1 - e)
    M_in = M6 * 1e6 * (1 + m_frac if Rs < rp0 else 1.0)
    s0 = initial_state(M_in, P, T0, e, inc, om, Om)
    tmin = min(t_ast_.min(), t_rv_.min()) - 0.1; tmax = max(t_ast_.max(), t_rv_.max()) + 0.1
    sols = []
    for tb in (tmin, tmax):
        if (tb - T0) == 0:
            continue
        sols.append(solve_ivp(rhs, (T0, tb), s0, method='DOP853', rtol=1e-11, atol=1e-9, dense_output=True,
                              args=(GM, Gm, Rs, d2)))

    def state(t):
        t = np.atleast_1d(t)
        out = np.empty((6, t.size))
        back = t < T0
        if back.any():
            out[:, back] = sols[0].sol(t[back])
        if (~back).any():
            out[:, ~back] = sols[1].sol(t[~back])
        return out

    def emit(t_obs):
        te = t_obs.copy()
        for _ in range(3):
            te = t_obs - state(te)[2] / C_AU
        return state(te)

    sa = emit(t_ast_)
    X = x0 + vx0 * (t_ast_ - TREF) + sa[0] / R0           # AU / kpc = mas
    Y = y0 + vy0 * (t_ast_ - TREF) + sa[1] / R0
    sv = emit(t_rv_)
    r = np.sqrt(np.sum(sv[:3] ** 2, axis=0)); v2 = np.sum(sv[3:] ** 2, axis=0)
    Phi = GM / r + np.where(r > Rs, Gm / r, Gm / Rs if np.isfinite(Rs) else 0.0)
    vz = sv[5] + (v2 / 2 + Phi) / C_AU                      # AU/yr, Upsilon = 1
    V = vz / KMS + vz0 + np.where(is_nirc2[:len(t_rv_)], dN, 0.0)
    return X, Y, V


BAD = 1e4 * np.ones(2 * len(t_ast) + len(t_rv))


def residuals(theta, m_frac=0.0, Rs=np.inf, free_m=False, d2=0.0):
    if free_m:
        m_frac = theta[-1]; theta = theta[:-1]
    if not (0 < theta[9] < 0.99 and theta[7] > 0 and theta[0] > 0 and theta[1] > 0 and m_frac > -0.5):
        return BAD
    X, Y, V = model(theta, m_frac, Rs, d2=d2)
    return np.concatenate([LXi @ (X - X_obs), LYi @ (Y - Y_obs), (V - V_obs) / sV])


def fit(theta0, Rs=np.inf, free_m=False, m0=0.0):
    p0 = np.concatenate([theta0, [m0]]) if free_m else np.array(theta0, float)
    res = least_squares(residuals, p0, args=(0.0, Rs, free_m), x_scale='jac', method='lm', max_nfev=4000)
    J = res.jac
    cov = np.linalg.pinv(J.T @ J)
    return res, cov


def jac_central(fun, p, steps):
    cols = []
    for k, h in enumerate(steps):
        dp = np.zeros_like(p); dp[k] = h
        cols.append((fun(p + dp) - fun(p - dp)) / (2 * h))
    return np.column_stack(cols)


def profiled_sigma_last(J, tol=1e-7):
    """sigma of the last parameter with all others profiled; inf if the profiled information is
    below tol times the unprofiled information (exact degeneracy up to finite-difference noise)."""
    s = np.linalg.norm(J, axis=0); Js = J / s
    F = Js.T @ Js
    Fm, Fmt, Ftt = F[-1, -1], F[-1, :-1], F[:-1, :-1]
    Ip = Fm - Fmt @ np.linalg.solve(Ftt, Fmt)
    ratio = Ip / Fm
    return (np.inf if ratio < tol else 1 / np.sqrt(Ip) / s[-1]), ratio


def upper95(m_hat, sig):
    """one-sided 95% profile-likelihood upper limit with the physical boundary m >= 0"""
    if not np.isfinite(sig):
        return np.inf
    return m_hat + 1.645 * sig if m_hat >= 0 else m_hat + np.sqrt(m_hat ** 2 + 2.706 * sig ** 2)


def fit_fixed_m(theta0, m, Rs):
    r = least_squares(lambda th: residuals(th, m, Rs), np.array(theta0, float), x_scale='jac', method='lm', max_nfev=4000)
    return r, float(r.fun @ r.fun)


def anomalistic(theta):
    M6, R0, x0, y0, vx0, vy0, vz0, P, T0, e, inc, om, Om, dN = theta
    GM = GM1 * M6 * 1e6
    s0 = initial_state(M6 * 1e6, P, T0, e, inc, om, Om)
    sol = solve_ivp(rhs, (T0, T0 + 1.5 * P), s0, method='DOP853', rtol=1e-12, atol=1e-10, dense_output=True,
                    args=(GM, 0.0, np.inf), events=lambda t, s, *a: s[:3] @ s[3:])
    tp = [t for t in sol.t_events[0] if t > T0 + 0.5 * P]
    t1 = tp[-1] if tp else np.nan
    x0v, x1v = sol.sol(T0)[:3], sol.sol(t1)[:3]
    ang = np.degrees(np.arccos(x0v @ x1v / np.linalg.norm(x0v) / np.linalg.norm(x1v))) * 60
    a = (M6 * 1e6 * P ** 2) ** (1 / 3)
    return t1 - T0, ang


if __name__ == '__main__':
    out = {}
    theta0 = [3.98, 7.92, 0.0, 0.0, 0.0, 0.0, 0.0, 16.04, 2018.378, 0.887, 133.4, 65.6, 226.3, 0.0]
    res, _ = fit(theta0)
    for _ in range(5):
        r_new, _ = fit(res.x)
        done = float(res.fun @ res.fun) - float(r_new.fun @ r_new.fun) < 1e-6
        res = r_new
        if done:
            break
    th = res.x; chi2 = float(res.fun @ res.fun); dof = res.fun.size - len(th)
    err0 = np.sqrt(np.diag(np.linalg.pinv(res.jac.T @ res.jac)))
    steps = 0.1 * err0
    J = jac_central(lambda p: residuals(p), th, steps)
    cov = np.linalg.inv(J.T @ J); err = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    print(f"no shell: chi2 = {chi2:.1f} for {dof} dof (reduced {chi2/dof:.2f})")
    for n, v, s_ in zip(NAMES, th, err):
        print(f"   {n:7s} = {v:12.5f} +- {s_:.5f}")
    Pan, dw = anomalistic(th)
    a_AU = (th[0] * 1e6 * th[7] ** 2) ** (1 / 3)
    print(f"   corr(M, R0) = {rho:.3f}; anomalistic period {Pan:.3f} yr; pericentre advance {dw:.2f} arcmin/orbit")
    print(f"   Do et al. 2019 GR chain: M = 3.951 +- 0.047, R0 = 7.935 +- 0.050 kpc, rho 0.972; NIRC2 offset 80.6 +- 19.1 km/s")
    lnth = np.log(th[0] / th[1]); lnth_c = np.log(3.95118 / 7.93494)
    print(f"   ln(M/R0) difference from the chain mean: {lnth - lnth_c:+.4f} (chain sd of ln theta_g 0.0060)")
    out['no_shell'] = dict(theta=th.tolist(), err=err.tolist(), chi2=chi2, dof=dof, corr_M_R0=rho,
                           anomalistic_period=Pan, periapse_advance_arcmin=dw)
    # orbit extent
    Ms = GM1 * th[0] * 1e6
    sol = solve_ivp(rhs, (th[8], th[8] + Pan), initial_state(th[0] * 1e6, *th[7:13]), method='DOP853',
                    rtol=1e-12, atol=1e-10, dense_output=True, args=(Ms, 0.0, np.inf))
    rr = np.linalg.norm(sol.sol(np.linspace(th[8], th[8] + Pan, 20001))[:3], axis=0)
    rp, ra = rr.min(), rr.max()
    AU_in_M = 23.56 * 4.3 / th[0]
    print(f"   pericentre {rp:.1f} AU = {rp*AU_in_M:.0f} M, apocentre {ra:.0f} AU = {ra*AU_in_M:.0f} M")
    out.update(rp_AU=rp, ra_AU=ra, AU_in_M=AU_in_M)

    # exact degeneracy of shells inside pericentre: fix m = 5% at 0.5 rp and refit
    th_start = th.copy(); th_start[0] = th[0] / 1.05
    rd, chi2d = fit_fixed_m(th_start, 0.05, 0.5 * rp)
    print(f"\ninside-pericentre test (m = 0.05 M at R = 0.5 rp, refit): delta chi2 = {chi2d - chi2:+.2e};"
          f" M_BH (1 + m) = {rd.x[0] * 1.05:.5f} vs no-shell M = {th[0]:.5f}")
    Jd = jac_central(lambda p: residuals(p[:-1], p[-1], 0.5 * rp), np.append(rd.x, 0.05), np.append(steps, 1e-3))
    sd, ratio = profiled_sigma_last(Jd)
    print(f"   profiled information ratio for m: {ratio:.1e} -> sigma(m) = {sd}")
    out['inside_test'] = dict(dchi2=chi2d - chi2, M_total=rd.x[0] * 1.05, M_noshell=th[0], info_ratio=ratio)

    grid = np.concatenate([np.logspace(np.log10(1.02 * rp), np.log10(0.98 * ra), 16), [1.3 * ra]])
    shells = []
    r0 = residuals(th)
    print("\nshell scan (m free, all 14 orbit/frame parameters profiled; linearised at the no-shell optimum,"
          " then polished with a full nonlinear refit):")
    for Rs in grid:
        f = lambda p, Rs=Rs: residuals(p[:-1], p[-1], Rs)
        J2 = jac_central(f, np.append(th, 0.0), np.append(steps, 1e-4))
        s_m, ratio = profiled_sigma_last(J2)
        if np.isfinite(s_m):
            dp = -np.linalg.lstsq(J2, r0, rcond=None)[0]           # Gauss-Newton step from (th, 0)
            m_lin = float(dp[-1])
            r2 = least_squares(f, np.append(th, 0.0) + dp, x_scale='jac', method='lm', max_nfev=4000)
            m_hat = float(r2.x[-1]); dchi = float(r2.fun @ r2.fun) - chi2
            theta_resp = -np.linalg.lstsq(J2[:, :-1], J2[:, -1], rcond=None)[0]   # d theta / d m at fixed m
        else:
            m_lin = m_hat = float('nan'); dchi = 0.0; theta_resp = None
        up = upper95(m_hat, s_m)
        shells.append(dict(R_AU=Rs, R_M=Rs * AU_in_M, m_hat=m_hat, m_lin=m_lin, sigma_m=s_m, upper95=up,
                           dchi2=dchi, dchi2_lin=-(m_lin / s_m) ** 2 if np.isfinite(s_m) else 0.0, info_ratio=ratio,
                           theta_resp=None if theta_resp is None else theta_resp.tolist()))
        print(f"   R = {Rs:7.1f} AU ({Rs*AU_in_M:6.0f} M): m = {m_hat:+.2e} (linear {m_lin:+.2e}) +- {s_m:.2e};"
              f" 95% upper (m >= 0) {up:.2e}; delta chi2 {dchi:+.2f} (linear {shells[-1]['dchi2_lin']:+.2f}); info ratio {ratio:.1e}")
        sys.stdout.flush()
    out['shells'] = shells
    # direct profile-likelihood check of the upper limits: refit with m fixed at m_up
    checks = []
    for k in [1, 6, 11]:
        sh = shells[k]; Rs = sh['R_AU']
        start = th + np.array(sh['theta_resp']) * sh['upper95']
        r_up, c_up = fit_fixed_m(start, sh['upper95'], Rs)
        c_min = chi2 + sh['dchi2'] if sh['m_hat'] >= 0 else chi2      # minimum within m >= 0
        checks.append(dict(R_AU=Rs, m_up=sh['upper95'], dchi2_at_up=c_up - c_min))
        print(f"   profile check R = {Rs:.0f} AU: chi2(m_up) - chi2(min over m >= 0) = {c_up - c_min:.2f} (target 2.71)")
        sys.stdout.flush()
    out['profile_checks'] = checks
    json.dump(out, open('results/s2_refit.json', 'w'), indent=1, default=float)
