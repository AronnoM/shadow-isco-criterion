"""Equation-level check of Psaltis et al. 2021 (arXiv:2012.02117) Eqs. 32-33 against the general kernel.
Shadow radius of a static metric depends on g_tt only: b = min_r r / sqrt(A(r)), A = -g_tt."""
import json
import numpy as np
from scipy.optimize import minimize_scalar


def shadow(A):
    f = lambda r: r / np.sqrt(A(r))
    return minimize_scalar(f, bounds=(2.5, 4.0), method='bounded', options={'xatol': 1e-12}).fun


b0 = 3 * np.sqrt(3); eps = 1e-5; out = {}
# (i) Eq. 9 truncated: g_tt = -1 + 2/r + 2 sum_i (-1)^i zeta_i / r^(i+1), only one zeta_i nonzero
for i in [1, 2, 3]:
    A = lambda r, i=i: 1 - 2 / r - 2 * (-1) ** i * eps / r ** (i + 1)
    c = (shadow(A) / b0 - 1) / eps
    out[f'zeta{i}_general'] = c
    print(f"Eq. 9 with only zeta_{i} = eps: d ln r_sh / d zeta_{i} = {c:+.6f} = {c*81:+.3f}/81"
          f"   (Psaltis Eq. 33: {[-19, 5, -1][i-1]}/81; general kernel -1/2 3^(1-n) delta_n: {[-27, 9, -3][i-1]}/81)")
# (ii) JP family (Eq. 18): g_tt = -(1 - 2/r)(1 + a12/r^2 + a13/r^3 + a14/r^4)^-2
for k, name in [(2, 'a12'), (3, 'a13'), (4, 'a14')]:
    A = lambda r, k=k: (1 - 2 / r) * (1 + eps / r ** k) ** -2
    c = (shadow(A) / b0 - 1) / eps
    out[f'{name}_JP'] = c
    print(f"JP with only {name} = eps: d ln r_sh / d {name} = {c:+.6f} = {c*81:+.3f}/81   (Psaltis Eq. 32: {[9, 3, 1][k-2]}/81)")
# (iii) Eq. 33 = Eq. 32 composed with the linearised JP map (Eq. 19) -> valid inside the JP family only,
# where the higher zeta_4, zeta_5, ... are fixed by a12-a14 (e.g. a14 also generates zeta_4).
print("Eq. 33 reproduces Eq. 32 through the JP map; for a metric in the Eq. 9 form truncated at zeta_3 the"
      " coefficients are -27/81, +9/81, -3/81.")
json.dump(out, open('results/psaltis_check.json', 'w'), indent=1)
