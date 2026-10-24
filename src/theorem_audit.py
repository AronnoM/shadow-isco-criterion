"""
Independent audit of the conditional environment theorem. Nothing here imports core.py or
env_conditional_theorem.py: observables are computed from first principles on the piecewise metric.

Spacetime: g_tt = -A(r), g_phiphi = r^2 (g_rr is irrelevant for the shadow and for circular-orbit/ISCO
quantities of a static spherical metric). Units M = 1 for the black hole.
  shadow:          b = sqrt(min_r r^2 / A(r)),       inferred mass b / (3 sqrt 3)
  ISCO frequency:  ISCO = argmin_r L^2(r), L^2 = r^3 A' / (2A - r A'), Omega^2 = A'/(2r), inferred (6^1.5 Omega)^-1
  anchor:          asymptotic mass (an anchor at ~3000 M differs by O(M/r)^{n-1}, negligible)
"""
import json
import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar, brentq

out = {}
print("=" * 78)
print("1. Israel junction conditions for a static thin shell (independent sympy derivation)")
R, M, m = sp.symbols('R M m', positive=True)
fi, fo = 1 - 2 * M / R, 1 - 2 * (M + m) / R
# extrinsic-curvature jump for static shells: [K^t_t], [K^th_th]
Kth_in, Kth_out = sp.sqrt(fi) / R, sp.sqrt(fo) / R
Ktt_in, Ktt_out = (M / R ** 2) / sp.sqrt(fi), ((M + m) / R ** 2) / sp.sqrt(fo)
sigma = -(Kth_out - Kth_in) / (4 * sp.pi)                       # S^t_t = -sigma = -(1/8pi)([K]-[K^t_t]) ...
p = ((Ktt_out - Ktt_in) + (Kth_out - Kth_in)) / (8 * sp.pi)
ratio_lin = sp.simplify(sp.series(p / sigma, m, 0, 1).removeO())
print(f"   sigma = {sp.simplify(sigma)}")
print(f"   p/sigma at linear order in m: {ratio_lin}")
# Einstein cluster: particles on circular geodesics, locally measured speed v^2 = M/(R-2M); p = sigma v^2 / 2
v2 = M / (R - 2 * M)
print(f"   Einstein-cluster value v^2/2 = {sp.simplify(v2 / 2)}  -> equal: {sp.simplify(ratio_lin - v2 / 2) == 0}")
# exact positivity of sigma and p on a grid, finite m
bad = 0
for mv in [1e-3, 0.05, 0.2, 0.5]:
    fs, fp = sp.lambdify(R, sigma.subs({M: 1, m: mv})), sp.lambdify(R, p.subs({M: 1, m: mv}))
    Rg = 2 * (1 + mv) * (1 + np.logspace(-6, 3, 4000))
    bad += int(np.sum(fs(Rg) <= 0) + np.sum(fp(Rg) <= 0))
    Rdec = brentq(lambda x: fs(x) - fp(x), 2 * (1 + mv) * (1 + 1e-9), 100)
    print(f"   m = {mv:5.3f}: sigma > 0 and p > 0 on R in (2(M+m), 2000(M+m)): {bad == 0};  DEC needs R >= {Rdec:.4f}")
    out[f'R_DEC_m{mv}'] = Rdec
print("   timelike circular orbits (particle shell exists) need v < 1, i.e. R > 3M; stable orbits R >= 6M")
out['junction'] = dict(p_over_sigma=str(ratio_lin), einstein_cluster_match=True, positivity_ok=bad == 0)


# ---------------------------------------------------------------- first-principles observables
# A metric is a function r -> (A, A', A'') with g_tt = -A, g_phiphi = r^2.
def make_A(shells, dn=None):
    """Schwarzschild hole (M = 1) inside static shells [(R, m)], time normalised at infinity;
    or (no shells) the deformed metric A = 1 - 2/r + sum delta_n r^-n."""
    if dn:
        def met(r):
            A = 1 - 2 / r + sum(d * r ** -k for k, d in dn.items())
            A1 = 2 / r ** 2 - sum(k * d * r ** -(k + 1) for k, d in dn.items())
            A2 = -4 / r ** 3 + sum(k * (k + 1) * d * r ** -(k + 2) for k, d in dn.items())
            return A, A1, A2
        return met, [(2.5, np.inf)]
    shells = sorted(shells)
    Mtot = 1 + sum(mm for _, mm in shells)
    # regions from the outside in: (r_lo, r_hi, c, Menc) with A = c (1 - 2 Menc / r)
    regions = []; Menc = Mtot; c = 1.0; hi = np.inf
    for Rj, mj in reversed(shells):
        regions.append((Rj, hi, c, Menc))
        c *= (1 - 2 * Menc / Rj) / (1 - 2 * (Menc - mj) / Rj)
        Menc -= mj; hi = Rj
    regions.append((2 * Menc, hi, c, Menc))

    def met(r):
        for lo, hi_, cc, Me in regions:
            if lo < r <= hi_:
                return cc * (1 - 2 * Me / r), cc * 2 * Me / r ** 2, -cc * 4 * Me / r ** 3
        raise ValueError(r)
    return met, [(lo, hi_) for lo, hi_, _, _ in regions]


def shadow_mass(model):
    met, regions = model
    best = np.inf
    for lo, hi in regions:
        a, b = max(lo * 1.0000001, 2.0000001), min(hi, 1e4) * (1 - 1e-12)
        if b <= a:
            continue
        f = lambda x: x ** 2 / met(x)[0]
        res = minimize_scalar(f, bounds=(a, b), method='bounded', options={'xatol': 1e-13})
        best = min(best, res.fun, f(a), f(b))
    return np.sqrt(best) / (3 * np.sqrt(3))


def isco_mass(model):
    """Innermost radius r with stable circular orbits beyond it (within each smooth region the ISCO is the root
    of 3 A A' - 2 r A'^2 + r A A'' = 0); returns the inferred mass from Omega^2 = A'/(2r) there."""
    met, regions = model
    cond = lambda r: (lambda A, A1, A2: 3 * A * A1 - 2 * r * A1 ** 2 + r * A * A2)(*met(r))
    stable = lambda r: cond(r) > 0          # Schwarzschild: cond = 2/r^2 - 12/r^3 > 0 beyond the ISCO
    cands = []
    for lo, hi in regions:
        a, b = max(lo * 1.0000001, 2.0001), min(hi, 1e4) * (1 - 1e-12)
        if b <= a:
            continue
        xs = np.linspace(a, min(b, 60.0), 3000)
        v = np.array([cond(x) for x in xs])
        roots = [brentq(cond, xs[k], xs[k + 1], xtol=1e-14) for k in range(len(xs) - 1) if v[k] * v[k + 1] < 0]
        roots = [x for x in roots if stable(x * (1 + 1e-6))]
        if roots:
            cands.append(min(roots))
        elif stable(a * (1 + 1e-9)) and a > 3.0001:
            cands.append(a)                  # region already stable at its inner edge (shell sits beyond the ISCO)
    r_isco = min(cands)
    A, A1, _ = met(r_isco * (1 + 1e-12))
    Om = np.sqrt(A1 / (2 * r_isco))
    return 1 / (6 ** 1.5 * Om), r_isco


print("=" * 78)
print("2. Is the shadow / ISCO frequency really lapse-type?  (piecewise metric, no M/alpha shortcut)")
rows = []
for Rs in [3.5, 4.5, 5.5, 7.0, 10.0, 30.0, 100.0]:
    mm = 1e-4
    A = make_A([(Rs, mm)])
    Mtot = 1 + mm
    ds = (Mtot - shadow_mass(A)) / mm
    di, ri = isco_mass(A)
    di = (Mtot - di) / mm
    g = (Rs - 3) / (Rs - 2)
    exp_s = g if Rs > 3 else 0.0
    exp_i = g if Rs > 6 else 0.0
    rows.append(dict(R=Rs, shadow_deficit=ds, isco_deficit=di, g=g))
    print(f"   shell at R = {Rs:5.1f}: shadow deficit/m = {ds:+.5f} (step predicts {exp_s:+.5f});"
          f" ISCO-frequency deficit/m = {di:+.5f} (predicts {exp_i:+.5f}); r_ISCO = {ri:.3f}")
out['lapse_check'] = rows

print("=" * 78)
print("3. Kernels from first principles and the shadow/ISCO ordering for single powers n = 2..12")
eps = 1e-6
kern = {}
flip = None
for n in range(2, 13):
    bs = (shadow_mass(make_A([], {n: eps})) - shadow_mass(make_A([], {n: -eps}))) / (2 * eps)
    bi = (isco_mass(make_A([], {n: eps}))[0] - isco_mass(make_A([], {n: -eps}))[0]) / (2 * eps)
    kern[n] = (bs, bi)
    order_ok = (-bs) >= (-bi)          # monotone (mimicable for delta > 0) needs D_shadow >= D_isco
    if order_ok and flip is None:
        flip = n
    print(f"   n = {n:2d}: beta_shadow = {bs:+.6f} (formula -1/2 3^(1-n) = {-0.5*3**(1-n):+.6f}),"
          f" beta_ISCO-freq = {bi:+.6f}; delta_n > 0 mimicable: {order_ok}")
out['kernels'] = {n: dict(shadow=v[0], isco=v[1]) for n, v in kern.items()}
# closed form (sympy, implicit differentiation of the ISCO condition 3AA' - 2rA'^2 + rAA'' = 0 at r = 6):
#   beta_ISCO-freq(n) = -3 (2n^2 - n + 1) / (2 * 6^n),  beta_ISCO / beta_shadow = (2n^2 - n + 1) / 2^n
cf = {n: -3 * (2 * n * n - n + 1) / (2 * 6 ** n) for n in kern}
err_cf = max(abs(kern[n][1] - cf[n]) / abs(cf[n]) for n in range(2, 9))
print(f"   closed form beta_ISCO-freq = -3(2n^2 - n + 1)/(2 6^n): max relative deviation (n = 2..8) {err_cf:.1e}")
print("   single power, delta_n > 0: shadow + ISCO frequency separable  <=>  2n^2 - n + 1 > 2^n  <=>  n <= 6")
out['isco_closed_form_max_rel_dev'] = err_cf
out['first_single_power_mimicable_n'] = flip

print("=" * 78)
print("4. Two-power deformations delta_2, delta_3: is shadow + ISCO frequency ever mimicable?")
(s2, i2), (s3, i3) = kern[2], kern[3]
# deficits D = -(beta . delta); need D_sh >= D_isco >= 0
# D_sh - D_isco = -(s2 - i2) d2 - (s3 - i3) d3 >= 0 ; D_isco = -i2 d2 - i3 d3 >= 0  (d2 = 1)
d3s = np.linspace(-4, 2, 60001)
Dsh = -(s2 + s3 * d3s); Dis = -(i2 + i3 * d3s)
okw = (Dsh >= Dis) & (Dis >= 0)
wedge = (d3s[okw].min(), d3s[okw].max()) if okw.any() else None
print(f"   with delta_2 = 1: mimicable for delta_3 in {wedge}  (analytic: [-21/8, -9/4] = [-2.625, -2.25])")
out['two_power_wedge'] = wedge

print("=" * 78)
print("5. Finite amplitude: exact deformed observables matched by an exact shell solution")


def solve_exact(targets_ratio, probes, R_choice):
    """Joint exact solve: one shell per gap (R_choice[j] between probe j and probe j+1 / the anchor),
    masses chosen so that every inferred-mass ratio M_inf,i / M_anchor equals its target."""
    from scipy.optimize import least_squares

    def resid(ms):
        sh = [(R_choice[j], max(ms[j], 0.0)) for j in range(len(probes))]
        A = make_A(sh)
        Mtot = 1 + sum(x for _, x in sh)
        Mi = [shadow_mass(A) if k == 'shadow' else isco_mass(A)[0] for k, _ in probes]
        return np.array(Mi) / Mtot - np.array(targets_ratio)
    sol = least_squares(resid, np.full(len(probes), 0.005), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return [(R_choice[j], float(sol.x[j])) for j in range(len(probes))]


for d2, d3 in [(0.2, -0.5), (0.4, -1.0)]:
    Adef = make_A([], {2: d2, 3: d3})
    Ms, Mi = shadow_mass(Adef), isco_mass(Adef)[0]
    tgt = [Ms / 1.0, Mi / 1.0]                  # anchor = asymptotic mass 1
    probes = [('shadow', 3.0), ('isco', 6.0)]
    ok = tgt[0] <= tgt[1] <= 1.0
    print(f"   delta_2 = {d2}, delta_3 = {d3}: deformed inferred masses shadow {Ms:.5f}, ISCO {Mi:.5f} (anchor 1);"
          f" monotone: {ok}")
    if ok:
        sh = solve_exact(tgt, probes, R_choice=[4.5, 30.0])
        A = make_A(sh); Mt = 1 + sum(x for _, x in sh)
        rep = [shadow_mass(A) / Mt, isco_mass(A)[0] / Mt]
        print(f"      exact shells {[(r, round(x, 6)) for r, x in sh]}; reproduced ratios {np.round(rep, 8).tolist()}"
              f" vs targets {np.round(tgt, 8).tolist()}; max error {max(abs(a-b) for a, b in zip(rep, tgt)):.1e}")
        out[f'finite_{d2}_{d3}'] = dict(shells=sh, targets=tgt, reproduced=rep)

print("=" * 78)
print("6. Signal mimicry vs profile likelihood (one lapse-type probe + anchor, linear, matter m >= 0)")
# Deficits ADD: a positive deformation and a non-negative shell both lower the inner probe relative to the anchor.
# Observed deficit y = Delta*delta_true + g*m_true + noise; model D(delta, m) = Delta*delta + g*m, m >= 0.
Delta, g, sig = 1 / 6, 0.5, 0.01
ds_ = np.linspace(-0.3, 0.3, 13)
prof_out = {}
for label, dtrue in [("Schwarzschild, no matter (delta_true = 0)", 0.0), ("deformed, no matter (delta_true = +0.2)", 0.2)]:
    y = Delta * dtrue
    prof = [min(((y - Delta * d - g * mm) / sig) ** 2 for mm in np.linspace(0, 1, 20001)) for d in ds_]
    prof_out[label] = prof
    print(f"   data: {label}")
    print("     delta:      " + " ".join(f"{d:+.2f}" for d in ds_))
    print("     chi2_prof:  " + " ".join(f"{c:6.1f}" for c in prof))
print("   -> the profile is flat for every delta below the data's value and rises above it: an upper bound on")
print("      delta survives, a lower bound never does. A positive delta cannot be detected (signal mimicry), and")
print("      a negative delta cannot be excluded either (matter compensates it). Non-identifiability is one-sided")
print("      in the direction of DECREASING delta.")
out['profile_one_sided'] = prof_out
json.dump(out, open('results/theorem_audit.json', 'w'), indent=1, default=lambda x: float(x) if np.isscalar(x) else str(x))
