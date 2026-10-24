"""Exact (finite-amplitude) two-shell configuration reproducing the exact flare and shadow masses of delta_2 = 0.5,
relative to a distant anchor (referee point 7: the linear mimic must be checked against exact observables)."""
import json
import numpy as np
from scipy.optimize import least_squares
from thin_shell import orbit_mass, alpha
from flare_consistent import flare_mass_def

R_E, R_IN, R_FAR, R_ANC, D2 = 9.0, 6.0, 1000.0, 2900.0, 0.5


def deformed():
    rp = (3 + np.sqrt(9 - 8 * D2)) / 2
    sh = (rp / np.sqrt(1 - 2 / rp + D2 / rp ** 2)) / (3 * np.sqrt(3))
    fl = flare_mass_def(R_E, 2, D2)
    anc = 1 - D2 / R_ANC                     # circular orbit at fixed areal radius, exact for n = 2
    return np.array([fl / anc, sh / anc])


def shells(m):
    m_in, m_far = m
    M1 = 1 + m_in                            # mass between the shells
    fl = M1 * orbit_mass(R_E / M1, m_far / M1, R_FAR / M1)
    a_in = alpha(m_in, R_IN)
    a_far = np.sqrt((1 - 2 * (M1 + m_far) / R_FAR) / (1 - 2 * M1 / R_FAR))
    sh = 1 / (a_in * a_far)
    anc = M1 + m_far
    return np.array([fl / anc, sh / anc])


if __name__ == '__main__':
    tgt = deformed()
    lin = json.load(open('results/flare_consistent.json'))['mimic_linear_d2_0.5']
    sol = least_squares(lambda m: shells(m) - tgt, [lin['m_in'], lin['m_far']], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    err = np.max(np.abs(shells(sol.x) - tgt))
    print(f"exact deformed ratios (flare, shadow)/anchor: {tgt}")
    print(f"linear mimic: {lin['m_in']*100:.2f}% at {lin['R_in']:.2f}M + {lin['m_far']*100:.2f}% at {lin['R_far']:.0f}M"
          f" -> exact ratios {shells([lin['m_in'], lin['m_far']])}")
    print(f"exact mimic : {sol.x[0]*100:.3f}% at {R_IN}M + {sol.x[1]*100:.3f}% at {R_FAR}M; max error {err:.1e}")
    json.dump(dict(target=tgt.tolist(), m_in=sol.x[0], m_far=sol.x[1], R_in=R_IN, R_far=R_FAR, err=err,
                   linear=lin), open('results/exact_mimic.json', 'w'), indent=1, default=float)
