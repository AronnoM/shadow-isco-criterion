"""
Partial identification of a deformation in the presence of non-negative shells (referee point 1 and 4A).

Linear model for the deficits relative to an exactly known anchor:  D = delta * Delta + S m + e,  m >= 0.
  separation      : the environment-only null reproduces delta*Delta iff delta*Delta in cone(S)       (Theorem 1)
  boundedness     : identified set I(D) = {delta : D - delta*Delta in cone(S)} is bounded below iff Delta not in
                    cone(S), above iff -Delta not in cone(S)
  point id.       : I(D) is a single point for every D iff Delta not in span(S)
Checks the referee's interior counterexample, the closed-form identified set for shadow + ISCO, and computes the
constrained profile likelihoods used in Fig. 5.
"""
import json
import numpy as np
from scipy.optimize import nnls
from core import kernel

g = lambda R: (R - 3.0) / (R - 2.0)


def deficit_template(n):
    return np.array([-kernel(n, 'shadow'), -kernel(n, 'isco_freq')])     # (D_sh, D_ISCO) per unit delta


def shell_resp(R):                                                       # deficit per unit m/M (shadow, ISCO)
    return np.array([g(R) if R > 3 else 0.0, g(R) if R > 6 else 0.0])


def identified_set(D, n, Rmin=3.0):
    """Closed form for shadow + ISCO with cone generators (1,0) [3 < R < 6] and (1,1) [R > 6]."""
    a = deficit_template(n)
    lo, hi = -np.inf, np.inf
    # D_I - delta a_I >= 0
    hi = min(hi, D[1] / a[1])
    if Rmin < 6:   # (D_sh - D_I) - delta (a_sh - a_I) >= 0
        c = a[0] - a[1]
        if c > 0:
            hi = min(hi, (D[0] - D[1]) / c)
        elif c < 0:
            lo = max(lo, (D[0] - D[1]) / c)
        return lo, hi
    # R >= 6M only: D - delta a must be proportional to (1, 1)
    c = a[0] - a[1]
    d0 = (D[0] - D[1]) / c
    return (d0, d0) if D[1] - d0 * a[1] >= -1e-15 else (np.nan, np.nan)


def profile(D, n, sig, cls, grid):
    """-2 ln L_p(delta) - min, with m >= 0 profiled over the environment class."""
    a = deficit_template(n); W = np.diag(1 / np.asarray(sig))
    gens = {'none': np.zeros((2, 0)), 'R>=6M': np.array([[1.0], [1.0]]),
            'R>=3M': np.array([[1.0, 1.0], [0.0, 1.0]])}[cls]
    out = []
    for d in grid:
        res = D - d * a
        if gens.shape[1] == 0:
            out.append(float(np.sum((W @ res) ** 2)))
        else:
            c, rn = nnls(W @ gens, W @ res)
            out.append(float(rn ** 2))
    out = np.array(out)
    return out - out.min()


if __name__ == '__main__':
    out = {}
    # ---- referee's counterexample
    Delta = deficit_template(2); s1, s2 = shell_resp(4.0), shell_resp(10.0)
    D = 1e-3 * s1 + 1e-3 * s2
    fam = []
    for t in np.linspace(-0.004, 0.003, 8):
        m1, m2 = 1e-3 + t / 4, 1e-3 - t / 3
        fam.append(float(np.max(np.abs(t * Delta + m1 * s1 + m2 * s2 - D))))
    lo, hi = identified_set(D, 2)
    print(f"counterexample: s1 = {s1}, s2 = {s2}, D = {D}; max |mean - D| along the family = {max(fam):.1e}")
    print(f"   identified set for delta_2: [{lo:.4f}, {hi:.4f}]  (8(D_I - D_sh) = {8*(D[1]-D[0]):.4f}, 24/7 D_I = {24/7*D[1]:.4f})")
    out['counterexample'] = dict(D=D.tolist(), family_max_err=max(fam), set=[lo, hi])
    # ---- closed form for general n
    rows = {}
    for n in range(2, 9):
        a = deficit_template(n)
        rows[n] = dict(a_sh=a[0], a_I=a[1], gap=a[1] - a[0])
        print(f"   n={n}: D_ISCO - D_sh per unit delta = {a[1]-a[0]:+.3e}" + (f" = 1/{1/(a[1]-a[0]):.0f}" if a[1] > a[0] else ""))
    out['gaps'] = rows
    # ---- profiles for the figure: deficits measured to 1% each, exactly known anchor
    sig = [0.01, 0.01]
    grid = np.linspace(-0.45, 0.25, 1401)
    truths = {'no matter': np.zeros(2), 'interior matter': 1e-2 * s1 + 1e-2 * s2}
    prof = {}
    for name, Dt in truths.items():
        prof[name] = {cls: profile(Dt, 2, sig, cls, grid).tolist() for cls in ['none', 'R>=6M', 'R>=3M']}
        for cls in ['none', 'R>=6M', 'R>=3M']:
            p = np.array(prof[name][cls]); inside = grid[p <= 3.84]
            print(f"   truth {name:15s}, environment {cls:6s}: 95% interval [{inside.min():+.4f}, {inside.max():+.4f}]")
    # asymmetry check (no matter, R >= 6M): curvature ratio of the two sides vs 1/sin^2(phi)
    a = deficit_template(2); s = np.ones(2); W = np.diag(1 / np.array(sig) ** 2)
    cphi = (a @ W @ s) / np.sqrt((a @ W @ a) * (s @ W @ s)); sphi = np.sqrt(1 - cphi ** 2)
    p = np.array(prof['no matter']['R>=6M']); q0 = np.array(prof['no matter']['none'])
    iL, iR = np.argmin(abs(grid + 0.05)), np.argmin(abs(grid - 0.05))
    print(f"   constrained profile, R>=6M, no matter: right side / unconstrained = {p[iR]/q0[iR]:.4f} (expect 1),"
          f" left side = {p[iL]/q0[iL]:.4f} (expect sin^2 phi = {sphi**2:.4f}; 1/sin phi = {1/sphi:.3f})")
    out.update(grid=grid.tolist(), profiles=prof, sigma=sig, sinphi=sphi)
    # ---- finite anchor: exact formula and the referee's example
    k = np.array([7 / 4, 1.0]); Wd = np.diag([4 / 7, 1.0])
    Q, P, R = k @ Wd @ k, k @ Wd @ s, s @ Wd @ s
    for aa in [0.0, 0.01, 0.1]:
        C = np.diag([7 / 4, 1.0]) + aa * np.ones((2, 2)); Ci = np.linalg.inv(C)
        c2 = (k @ Ci @ s) ** 2 / ((k @ Ci @ k) * (s @ Ci @ s))
        c2f = P ** 2 / (R * Q + aa * R * (Q * R - P ** 2))
        print(f"   anchor variance a = {aa}: 1/sin phi = {1/np.sqrt(1-c2):.4f} (formula {1/np.sqrt(1-c2f):.4f})")
    json.dump(out, open('results/partial_id.json', 'w'), indent=1, default=float)
