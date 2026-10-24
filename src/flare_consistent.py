"""
Flare astrometry with ONE estimator for both hypotheses (referee point 3).

Observable: face-on circular orbit at areal radius r_e; measured = (image radius b_obs, Omega_obs at infinity).
Estimator: the Schwarzschild pair (M_fit, r_fit) with b_Schw(r_fit, M_fit) = b_obs and M_fit / r_fit^3 = Omega_obs^2.
The same estimator is applied to (i) the deformed metric A = B = 1 - 2/r + delta r^-n and (ii) the
thin-shell spacetime (thin_shell.orbit_mass). Units M = 1.

Outputs: deformation kernels from the astrometric fit vs the fixed-radius formula, cone angles for the
probe sets used in the paper, and the linear two-shell mimic of delta_2 = 0.5.
"""
import json
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, nnls
from thin_shell import faceon_b_schw, faceon_r_schw, orbit_mass, shadow_mass, isco_mass
from core import kernel

R_E = 9.0
R_ANCHOR = 2900.0


def sweep_def(b, n, d, u_hi):
    f = lambda u: 1.0 / np.sqrt(1.0 / b ** 2 - u * u * (1 - 2 * u + d * u ** n))
    return quad(f, 0.0, u_hi, epsabs=1e-14, epsrel=1e-13, limit=400)[0]


def faceon_b_def(r, n, d):
    A = 1 - 2 / r + d * r ** -n
    bmax = r / np.sqrt(A) * (1 - 1e-12)
    return brentq(lambda b: sweep_def(b, n, d, 1 / r) - np.pi / 2, 0.5 * r, bmax, xtol=1e-15)


def fit_schw(b, Om2):
    Mi = 1.0
    for _ in range(80):
        Mn = Om2 * faceon_r_schw(b, Mi) ** 3
        if abs(Mn - Mi) < 1e-15:
            break
        Mi = Mn
    return Mi


def flare_mass_def(r, n, d):
    Om2 = (2 / r ** 2 - n * d * r ** -(n + 1)) / (2 * r)      # A'/(2r)
    return fit_schw(faceon_b_def(r, n, d), Om2)


def flare_kernel(n, r=R_E, h=1e-4):
    return (np.log(flare_mass_def(r, n, h)) - np.log(flare_mass_def(r, n, -h))) / (2 * h)


if __name__ == '__main__':
    out = {}
    print("deformation kernels of the face-on flare at 9M (same astrometric estimator as the shell response):")
    kern = {}
    for n in [2, 3, 4]:
        ks = [flare_kernel(n, h=h) for h in (1e-3, 1e-4, 1e-5)]
        kern[n] = ks[1]
        print(f"  n={n}: astrometric fit {ks[0]:+.10f} {ks[1]:+.10f} {ks[2]:+.10f}   fixed radius {kernel(n,'orbit',R_E):+.10f}")
    out['flare_kernel_astrometric'] = kern
    out['flare_kernel_fixed_radius'] = {n: kernel(n, 'orbit', R_E) for n in kern}

    # shell responses, relative to the anchor (anchor outside all shells sees M + m)
    Rg = np.unique(np.concatenate([np.linspace(2.6, 8.98, 120), np.linspace(9.02, 30, 120),
                                   np.logspace(np.log10(30), np.log10(R_ANCHOR * 0.999), 120)]))
    dm = 1e-6
    resp = []
    for R in Rg:
        fl = (orbit_mass(R_E, dm, R) - (1 + dm)) / dm
        sh = (shadow_mass(dm, R) - (1 + dm)) / dm
        isc = (isco_mass(dm, R) - (1 + dm)) / dm
        resp.append((R, fl, sh, isc))
    resp = np.array(resp)
    i12 = np.argmin(abs(resp[:, 0] - 12))
    print(f"  shell at R={resp[i12,0]:.2f}: flare {resp[i12,1]:+.5f}, shadow {resp[i12,2]:+.5f}, ISCO {resp[i12,3]:+.5f}")
    out['shell_resp_R12'] = resp[i12].tolist()

    # deficit templates (relative to anchor, per unit delta): D = beta_anchor - beta_probe
    def templ(n, probes):
        ba = kernel(n, 'orbit', R_ANCHOR)
        k = {'flare': kern[n], 'shadow': kernel(n, 'shadow'), 'isco': kernel(n, 'isco_freq')}
        return np.array([ba - k[p] for p in probes])

    col = {'flare': 1, 'shadow': 2, 'isco': 3}

    def env_matrix(probes, mask):
        # deficit produced by a shell = -(response relative to anchor)
        return -np.column_stack([[r_[col[p]] for p in probes] for r_ in resp[mask]])

    def max_cos(u, S, sig):
        W = np.diag(1 / np.asarray(sig) ** 2); Wh = np.sqrt(W)
        U = Wh @ u; Sm = Wh @ S
        c, res = nnls(Sm, U)
        return np.sqrt(max(0.0, 1 - res ** 2 / (U @ U))), c

    # precise anchor: the deficits are measured directly with the probe errors
    sig = {'flare': 0.40, 'shadow': 0.20, 'isco': 0.20}
    sets = {'flare+shadow': ['flare', 'shadow'], 'ISCO+shadow': ['isco', 'shadow'],
            'flare+ISCO+shadow': ['flare', 'isco', 'shadow']}
    cones = {}
    for name, pr in sets.items():
        for n in [2, 3]:
            u = templ(n, pr)
            row = {}
            for Rmin in [2.6, 3.2, 6.0, 9.0]:
                mask = resp[:, 0] >= Rmin
                c, _ = max_cos(u, env_matrix(pr, mask), [sig[p] for p in pr])
                row[Rmin] = c
            cones[f'{name} n={n}'] = row
            print(f"  {name:18s} n={n}: deficits {np.round(u,5)}  max cos over cone, R>=2.6/3.2/6/9: "
                  + " ".join(f"{row[k]:.4f}" for k in row))
    out['cones'] = cones

    # linear two-shell mimic of delta_2 = 0.5 for flare+shadow: one shell in (3, 9), one far out
    d2 = 0.5
    tgt = d2 * templ(2, ['flare', 'shadow'])
    best = None
    for R_in in [3.2, 4.0, 5.0, 6.0, 7.0, 8.0]:
        i_in = np.argmin(abs(resp[:, 0] - R_in)); i_far = np.argmin(abs(resp[:, 0] - 1000))
        Smat = -np.array([[resp[i_far, 1], resp[i_in, 1]], [resp[i_far, 2], resp[i_in, 2]]])
        m = np.linalg.solve(Smat, tgt)
        if np.all(m >= 0):
            tot = m.sum()
            if best is None or abs(resp[i_in, 0] - 6.0) < abs(best['R_in'] - 6.0):
                best = dict(R_far=resp[i_far, 0], m_far=m[0], R_in=resp[i_in, 0], m_in=m[1], total=tot)
            print(f"  mimic d2=0.5 (linear): {m[0]*100:.2f}% at R={resp[i_far,0]:.0f}M + {m[1]*100:.2f}% at R={resp[i_in,0]:.2f}M")
    out['mimic_linear_d2_0.5'] = best
    # exact shadow deficit at d2 = 0.5 vs linear
    rp = (3 + np.sqrt(9 - 8 * d2)) / 2
    A = 1 - 2 / rp + d2 / rp ** 2
    out['shadow_deficit_exact_d2_0.5'] = 1 - (rp / np.sqrt(A)) / (3 * np.sqrt(3))
    out['flare_deficit_exact_d2_0.5'] = 1 - flare_mass_def(R_E, 2, d2)
    print(f"  d2=0.5 exact deficits: shadow {out['shadow_deficit_exact_d2_0.5']:.5f} (linear {d2/6:.5f}),"
          f" flare {out['flare_deficit_exact_d2_0.5']:.5f} (linear {-d2*kern[2]:.5f})")
    json.dump(out, open('results/flare_consistent.json', 'w'), indent=1, default=float)


# ---------------------------------------------------------------- radial-only (regular) family: A = 1 - 2/r,
# B = (1 - 2/r)(1 + gamma r^-n). Omega is unchanged; the image radius is not (photons feel g_rr).
def sweep_radial(b, n, gm, r_e):
    # phi swept from r_e to infinity: int b sqrt(A/B) / (r^2 sqrt(1 - A b^2/r^2)) dr, in u = 1/r
    f = lambda u: 1.0 / np.sqrt(1 + gm * u ** n) / np.sqrt(1.0 / b ** 2 - u * u * (1 - 2 * u))
    return quad(f, 0.0, 1 / r_e, epsabs=1e-14, epsrel=1e-13, limit=400)[0]


def flare_mass_radial(r, n, gm):
    bmax = r / np.sqrt(1 - 2 / r) * (1 - 1e-12)
    b = brentq(lambda b: sweep_radial(b, n, gm, r) - np.pi / 2, 0.5 * r, bmax, xtol=1e-15)
    return fit_schw(b, 1.0 / r ** 3)


def flare_kernel_radial(n, r=R_E, h=1e-4):
    return (np.log(flare_mass_radial(r, n, h)) - np.log(flare_mass_radial(r, n, -h))) / (2 * h)
