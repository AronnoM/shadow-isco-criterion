"""
Conditional environment theorem: explicit hypotheses, energy conditions, constructive proof checks.

Hypotheses (H1-H5), linear order in the deformation amplitude delta and in the shell masses:
 H1 Spacetime family: static, spherically symmetric; Schwarzschild black hole of mass M surrounded by
    finitely many static thin shells (m_j >= 0, radii R_j), each satisfying the energy conditions below.
 H2 Deformation family: A(r) = 1 - 2M/r + sum_n delta_n (M/r)^n, g_rr = 1/A, n >= 2, delta_n > 0.
 H3 Probes: an anchor outside all shells and K "lapse-type" probes whose observable is either a frequency
    (measured with the asymptotic time) of a geodesic process confined inside radius r_i, or the critical
    impact parameter of a photon sphere at r_i; each is inverted for a mass assuming Schwarzschild.
    (Shadow: r = 3M. ISCO orbital frequency: r = 6M. Astrometric orbits are NOT lapse-type: excluded.)
 H4 Each sector reports one mass (mass-only summaries), linear-Gaussian regime.
 H5 Admissible shell radii R >= R_adm, with R_adm set by the chosen energy condition (below).

Claim (linear order): deficits Delta_i = beta_anchor - beta_i of the deformation (per unit delta) are reproduced
exactly by some admissible non-negative shell configuration  <=>  (a) Delta is non-negative and non-increasing
in probe radius, and (b) every gap (max(r_j, R_adm), r_{j+1}) in which Delta drops contains an admissible radius.
"""
import json
import numpy as np
import sympy as sp
from scipy.optimize import nnls, brentq
from core import kernel

# ------------------------------------------------------------------ exact nested-shell responses
def inferred_masses(M0, shells, probes):
    """shells: list of (R, m); probes: list of (kind, r) with kind in {'lapse','anchor'}.
    Lapse-type probe at r: M_enc(r) / prod_{R_j > r} alpha_j. Anchor (outside all): total mass."""
    shells = sorted(shells)
    Mins, Mouts, alphas = [], [], []
    Menc = M0
    for Rj, mj in shells:
        Mins.append(Menc); Mouts.append(Menc + mj)
        alphas.append(np.sqrt((1 - 2 * (Menc + mj) / Rj) / (1 - 2 * Menc / Rj)))
        Menc += mj
    res = []
    for kind, r in probes:
        if kind == 'anchor':
            res.append(Menc); continue
        Mloc = M0 + sum(mj for Rj, mj in shells if Rj < r)
        lapse = np.prod([a for (Rj, mj), a in zip(shells, alphas) if Rj > r]) if shells else 1.0
        res.append(Mloc / lapse)
    return np.array(res)


def construct(probe_r, Delta, delta, R_adm=2.5):
    """Linear-order construction: one shell per gap where Delta drops."""
    K = len(probe_r); Dext = list(Delta) + [0.0]
    edges = list(probe_r) + [2800.0]
    shells = []
    for j in range(K):
        drop = Dext[j] - Dext[j + 1]
        if drop <= 0:
            continue
        lo, hi = max(edges[j], R_adm, 3.0), edges[j + 1]
        if hi <= lo:
            raise ValueError("gap has no admissible radius")
        Rj = np.sqrt(lo * hi) if hi / lo > 1.5 else 0.5 * (lo + hi)
        g = (Rj - 3) / (Rj - 2)
        shells.append((Rj, drop * delta / g))
    return shells


def shell_template(r_probes, Rj):
    return np.array([(Rj - 3) / (Rj - 2) if ri < Rj else 0.0 for ri in r_probes])


if __name__ == '__main__':
    out = {}

    # ------------------------------------------------------------------ energy conditions (Israel junction)
    R, M, m = sp.symbols('R M m', positive=True)
    fin, fout = 1 - 2 * M / R, 1 - 2 * (M + m) / R
    sigma = (sp.sqrt(fin) - sp.sqrt(fout)) / (4 * sp.pi * R)
    p = ((1 - (M + m) / R) / sp.sqrt(fout) - (1 - M / R) / sp.sqrt(fin)) / (8 * sp.pi * R)
    ratio_lin = sp.simplify(sp.limit(p / sigma, m, 0))
    print(f"Israel junction, static thin shell: p/sigma -> {ratio_lin} as m -> 0")
    out['p_over_sigma_linear'] = str(ratio_lin)
    # exact DEC boundary (sigma = p) for finite m, units M = 1
    for mv in [1e-4, 0.01, 0.05, 0.1]:
        g = sp.lambdify(R, (sigma - p).subs({M: 1, m: mv}))
        Rdec = brentq(g, 2 * (1 + mv) + 1e-9, 50)
        print(f"  m = {mv:6.4f} M: sigma > 0 for all R > 2(M+m); p > 0; DEC (p <= sigma) requires R >= {Rdec:.4f} M")
        out[f'R_DEC_m{mv}'] = Rdec
    out['admissibility'] = {'WEC/NEC/SEC': 'R > 2(M+m)', 'DEC': '~2.5 M (linear)', 'stable Einstein cluster': '6 M'}

    rng = np.random.default_rng(5)
    errs, negs = [], []
    for trial in range(400):
        K = rng.integers(1, 5)
        r = np.sort(rng.uniform(3.0, 60.0, K))
        Delta = np.sort(rng.uniform(0, 0.4, K))[::-1]                 # non-increasing, non-negative
        delta = 1e-4
        shells = construct(r, Delta, delta)
        probes = [('lapse', ri) for ri in r] + [('anchor', None)]
        Minf = inferred_masses(1.0, shells, probes)
        deficit = (Minf[-1] - Minf[:-1]) / delta                       # per unit delta, relative to anchor
        errs.append(np.max(np.abs(deficit - Delta)))
    print(f"\nconstruction check (400 random monotone deficit vectors, exact nested-shell spacetime, delta = 1e-4):"
          f"\n  max |reproduced - target| = {max(errs):.2e}  (linear-order residual O(delta))")
    out['construction_max_err'] = float(max(errs))


    # converse: non-monotone deficits cannot be reached by any non-negative shell set (dense grid, NNLS)
    res_conv = []
    for trial in range(400):
        K = rng.integers(2, 5)
        r = np.sort(rng.uniform(3.0, 60.0, K))
        Delta = rng.uniform(0, 0.4, K)
        grid = np.concatenate([np.linspace(3.01, 60, 600), np.logspace(np.log10(60), np.log10(2800), 200)])
        S = np.column_stack([shell_template(r, Rj) for Rj in grid])
        c, resid = nnls(S, Delta)
        monotone = bool(np.all(np.diff(Delta) <= 0))
        res_conv.append((monotone, resid / np.linalg.norm(Delta)))
    mono = [x[1] for x in res_conv if x[0]]; non = [x[1] for x in res_conv if not x[0]]
    print(f"converse check: monotone targets -> max relative NNLS residual {max(mono) if mono else float('nan'):.1e} ({len(mono)} cases);"
          f" non-monotone -> min residual {min(non):.2e} ({len(non)} cases)")
    out['converse'] = dict(monotone_max_resid=max(mono) if mono else None, nonmonotone_min_resid=min(non), n_non=len(non))

    # ------------------------------------------------------------------ which deformation/probe sets satisfy (a)
    print("\nhypothesis (a) for H2 deformations with lapse-type probes (deficits relative to an anchor with beta ~ 0):")
    sets = {'shadow (3M) only': [('shadow', 3.0)], 'ISCO frequency (6M) only': [('isco_freq', 6.0)],
            'shadow (3M) + ISCO frequency (6M)': [('shadow', 3.0), ('isco_freq', 6.0)]}
    tab = {}
    for name, probes in sets.items():
        for n in [2, 3, 4]:
            D = [-kernel(n, k) for k, _ in probes]
            ok = all(D[i] >= D[i + 1] for i in range(len(D) - 1)) and min(D) >= 0
            tab[f"{name} n={n}"] = dict(deficits=D, mimicable=ok)
            print(f"  {name:36s} n={n}: deficits {np.round(D, 4).tolist()} -> {'mimicable' if ok else 'NOT mimicable'}")
    out['hypothesis_a'] = tab
    json.dump(out, open('results/env_conditional_theorem.json', 'w'), indent=1, default=float)
