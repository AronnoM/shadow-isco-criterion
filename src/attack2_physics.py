"""
Attack 2 -- physics stress tests (units M = 1, metric A = 1 - 2/r + d (1/r)^n, g_rr = 1/A).
 (i)   nonlinearity: exact mass inferred by each probe vs the linear kernel
 (ii)  finite-l ringdown: WKB3 QNM kernels vs the eikonal / shadow kernel (blind-pair leakage)
 (iii) environment vs gravity: extended-mass template vs deformation templates (Sgr A*-like)
"""
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from core import kernel, confusion, sigma_eps
from wkb import qnm_wkb3

out = {}
A = lambda r, d, n: 1 - 2 / r + d * r ** (-n)
Ap = lambda r, d, n: 2 / r ** 2 - n * d * r ** (-n - 1)


# ------------------------------------------------------------- exact probe responses
def M_orbit(d, n, r):            # orbit with r and Omega both measured: M = r^2 A'/2  (exact)
    return r ** 2 * Ap(r, d, n) / 2


def M_shadow(d, n):              # b_c = r_c / sqrt(A(r_c)); Schwarzschild b_c = 3 sqrt3 M
    res = minimize_scalar(lambda r: -A(r, d, n) / r ** 2, bounds=(1.5, 6), method='bounded',
                          options=dict(xatol=1e-12))
    return 1 / np.sqrt(A(res.x, d, n) / res.x ** 2) / (3 * np.sqrt(3))


def M_isco(d, n):                # orbital frequency at ISCO; Schwarzschild Omega = 1/(6 sqrt6 M)
    L2 = lambda r: r ** 3 * Ap(r, d, n) / (2 * A(r, d, n) - r * Ap(r, d, n))
    h = 1e-6
    dL2 = lambda r: (L2(r + h) - L2(r - h)) / (2 * h)
    ri = brentq(dL2, 3.2, 12)
    Om = np.sqrt(Ap(ri, d, n) / (2 * ri))
    return 1 / (6 * np.sqrt(6) * Om)


print("(i) nonlinearity: (M_inf/M - 1) exact vs linear kernel * d")
res_i = {}
for n in [2, 3]:
    rows = []
    for d in [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 0.8]:
        try:
            ex_sh = M_shadow(d, n) - 1; ex_is = M_isco(d, n) - 1
        except ValueError:
            continue
        rows.append(dict(d=d, shadow_exact=ex_sh, shadow_lin=kernel(n, 'shadow') * d,
                         isco_exact=ex_is, isco_lin=kernel(n, 'isco_freq') * d,
                         orbit9_exact=M_orbit(d, n, 9.0) - 1, orbit9_lin=kernel(n, 'orbit', 9.0) * d))
    res_i[n] = rows
    for row in rows:
        print(f"  n={n} d={row['d']:+.2f}: shadow {row['shadow_exact']:+.4f} vs {row['shadow_lin']:+.4f}"
              f" | ISCO {row['isco_exact']:+.4f} vs {row['isco_lin']:+.4f}"
              f" | orbit@9M {row['orbit9_exact']:+.4f} vs {row['orbit9_lin']:+.4f}")
out['i_nonlinear'] = res_i

# leakage of a finite deformation into T_perp (Sgr A*-like: S2, flare orbit 9M, EHT shadow)
print("\n    leakage of finite-d signal into T_perp (S2, flare@9M, EHT; sigmas 0.01, 0.10, 0.09)")
Amat = np.ones((3, 1)); C = np.diag([0.01, 0.10, 0.09]) ** 2
leak = []
for n in [2, 3]:
    u = np.array([kernel(n, 'orbit', 2800.0), kernel(n, 'orbit', 9.0), kernel(n, 'shadow')])
    for d in [0.1, 0.3, 0.5, 0.8]:
        m = np.array([M_orbit(d, n, 2800.0) - 1, M_orbit(d, n, 9.0) - 1, M_shadow(d, n) - 1])
        cos = confusion(Amat, C, u, m)
        leak.append(dict(n=n, d=d, frac_noncentrality_in_perp=1 - cos ** 2))
        print(f"      n={n} d={d:.1f}: fraction of signal in T_perp = {1 - cos**2:.2e}")
out['i_leakage'] = leak

# ------------------------------------------------------------- (ii) finite-l ringdown
print("\n(ii) ringdown kernels from WKB3 vs shadow kernel (beta = dM_inf/(M dd))")
eps = 1e-4
res_ii = []
for n in [2, 3]:
    ksh = kernel(n, 'shadow'); kwI = kernel(n, 'eikonal_wI')
    for s in [0, 1, 2]:
        for l in ([2, 3, 4, 6, 10] if s != 1 else [1, 2, 3, 6, 10]):
            if s == 2 and l < 2:
                continue
            wp = qnm_wkb3(eps, n=n, l=l, s=s); wm = qnm_wkb3(-eps, n=n, l=l, s=s)
            bR = -(np.log(wp.real) - np.log(wm.real)) / (2 * eps)
            bI = -(np.log(-wp.imag) - np.log(-wm.imag)) / (2 * eps)
            res_ii.append(dict(n=n, s=s, l=l, beta_R=bR, beta_I=bI, beta_shadow=ksh, beta_eik_I=kwI,
                               leak_R=bR - ksh))
            print(f"  n={n} s={s} l={l:2d}: beta_R={bR:+.4f} (shadow {ksh:+.4f}, diff {bR-ksh:+.4f})"
                  f"   beta_I={bI:+.4f} (eikonal {kwI:+.4f})")
out['ii_ringdown'] = res_ii

# ------------------------------------------------------------- (iii) environment vs gravity
print("\n(iii) extended mass vs metric deformation (summary space, mass-only sectors)")
# sectors: GRAVITY-S2, Keck-S2 (r~2800M), flare (9M), EHT shadow. Fractional sigmas:
names = ['GRAVITY', 'Keck', 'flare', 'EHT']
sig = np.array([0.0043, 0.0064, 0.40, 0.20])
Am = np.ones((4, 1)); Cm = np.diag(sig ** 2)
res_iii = []
for gamma in [1.5, 2.0, 2.5]:
    r0 = 2800.0
    env = np.array([1.0, 1.0, (9.0 / r0) ** (3 - gamma), 0.0])      # fractional M_enc excess, unit at S2
    for n in [2, 3]:
        u = np.array([kernel(n, 'orbit', 2800.0)] * 2 + [kernel(n, 'orbit', 9.0), kernel(n, 'shadow')])
        cos = confusion(Am, Cm, u, env)
        s0 = sigma_eps(Am, Cm, u); s1 = sigma_eps(Am, Cm, u, [env])
        res_iii.append(dict(gamma=gamma, n=n, cos=cos, sigma_d=s0, sigma_d_env_marg=s1))
        print(f"  gamma={gamma} n={n}: |cos(env, d_n)| = {cos:.4f}; sigma(d_n) {s0:.3f} -> {s1:.3f} with env marginalised")
# what flare precision would separate them?
print("  flare precision needed (gamma=2, n=2):")
for sf in [0.40, 0.10, 0.05, 0.02, 0.01]:
    sg = sig.copy(); sg[2] = sf; Cm2 = np.diag(sg ** 2)
    u = np.array([kernel(2, 'orbit', 2800.0)] * 2 + [kernel(2, 'orbit', 9.0), kernel(2, 'shadow')])
    env = np.array([1.0, 1.0, (9.0 / 2800.0), 0.0])
    cos = confusion(Am, Cm2, u, env)
    print(f"    sigma_flare={sf:.2f}: |cos|={cos:.4f}, inflation 1/sin = {1/np.sqrt(1-cos**2):.2f}")
    res_iii.append(dict(flare_sigma=sf, cos=cos))
out['iii_environment'] = res_iii

json.dump(out, open('results/attack2.json', 'w'), indent=1, default=float)
