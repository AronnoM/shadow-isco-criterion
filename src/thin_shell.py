"""
Thin static shell (mass m, radius R) around a Schwarzschild black hole (mass M). Units M = 1.

Interior 2 < r < R :  ds^2 = -alpha^2 (1-2/r) dt^2 + dr^2/(1-2/r) + r^2 dOmega^2
Exterior r > R     :  ds^2 = -(1-2(1+m)/r) dt^2 + dr^2/(1-2(1+m)/r) + r^2 dOmega^2
Lapse normalisation (t = proper time at infinity, g_tt continuous at R):
    alpha^2 = (1 - 2(1+m)/R) / (1 - 2/R)
Null geodesics conserve E = -p_t and L = p_phi globally, so b = L/E is continuous across the shell;
inside, rays behave as Schwarzschild(M=1) rays with local impact parameter alpha*b.

What a distant observer infers when modelling the object as Schwarzschild(M_inf):
  shadow        : b_c = 3 sqrt3 / alpha  (photon sphere inside the shell, R > 3)  ->  M_inf = 1/alpha
  ISCO frequency: Omega_inf = alpha/(6 sqrt6)                                     ->  M_inf = 1/alpha
  orbit at r_f with astrometric radius (face-on, ray-traced) and Omega_inf = alpha sqrt(1/r_f^3):
                  solve self-consistently for M_inf (function orbit_mass below)
  anchor (S2, r >> R): Kepler in the exterior                                     ->  M_inf = 1 + m
If the shell lies inside a probe's radius, that probe sees the exterior metric -> M_inf = 1 + m.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def alpha(m, R):
    return np.sqrt((1 - 2 * (1 + m) / R) / (1 - 2 / R))


def _sweep(bloc, Mloc, u_lo, u_hi):
    """phi swept by an outgoing photon between u_lo < u_hi in Schwarzschild(Mloc), impact bloc.
    Substitution u = u_hi - (u_hi-u_lo) s^2 is not needed: no turning point for direct rays."""
    f = lambda u: 1.0 / np.sqrt(1.0 / bloc ** 2 - u * u * (1 - 2 * Mloc * u))
    val, _ = quad(f, u_lo, u_hi, epsabs=1e-13, epsrel=1e-12, limit=200)
    return val


def faceon_b_schw(r, Mi):
    """Apparent radius (impact parameter) of a face-on ring of radius r in Schwarzschild(Mi)."""
    g = lambda b: _sweep(b, Mi, 0.0, 1.0 / r) - np.pi / 2
    bmax = r / np.sqrt(1 - 2 * Mi / r) * (1 - 1e-12)
    return brentq(g, 0.5 * r, bmax, xtol=1e-14)


def faceon_r_schw(b, Mi):
    """Inverse of faceon_b_schw at fixed Mi."""
    return brentq(lambda r: faceon_b_schw(r, Mi) - b, 3.05 * Mi, 50 * b, xtol=1e-13)


def faceon_b_shell(r, m, R):
    """Apparent radius of a face-on ring at r < R in the shell spacetime (ray-traced)."""
    a = alpha(m, R)
    def g(b):
        return _sweep(a * b, 1.0, 1.0 / R, 1.0 / r) + _sweep(b, 1.0 + m, 0.0, 1.0 / R) - np.pi / 2
    # bracket: rays must stay below the local critical values in both regions
    bmax_in = r / np.sqrt(1 - 2 / r) / a
    return brentq(g, 0.3 * r, bmax_in * (1 - 1e-12), xtol=1e-14)


def orbit_mass(r, m, R):
    """Mass inferred from a circular orbit at r (astrometric radius + Omega at infinity)."""
    if R <= r:                                    # shell inside the orbit: exterior Schwarzschild(1+m)
        return 1.0 + m
    a = alpha(m, R)
    Om2 = a ** 2 / r ** 3                          # dphi/dt with global t
    b = faceon_b_shell(r, m, R)
    Mi = 1.0
    for _ in range(60):                            # fixed point M_i = Omega^2 r_i(b, M_i)^3
        Mn = Om2 * faceon_r_schw(b, Mi) ** 3
        if abs(Mn - Mi) < 1e-14:
            break
        Mi = Mn
    return Mi


def shadow_mass(m, R):
    if R <= 3 * (1 + m):                          # exterior photon sphere exists -> sees 1 + m
        return 1.0 + m
    return 1.0 / alpha(m, R)


def isco_mass(m, R):
    if R <= 6 * (1 + m):
        return 1.0 + m
    return 1.0 / alpha(m, R)


def anchor_mass(m, R, r_anchor=2800.0):
    return 1.0 + m if R < r_anchor else 1.0 / alpha(m, R)


if __name__ == "__main__":
    # sanity: no shell -> every probe infers M = 1
    print("m=0 checks:", orbit_mass(9.0, 0.0, 50.0), shadow_mass(0.0, 50.0))
    for R in [6.0, 8.0, 12.0, 30.0, 100.0, 1000.0]:
        m = 1e-4
        resp = {k: (f - (1 + m)) / m for k, f in
                [('flare9', orbit_mass(9.0, m, R)), ('shadow', shadow_mass(m, R))]}
        lin = -(R - 3) / (R - 2)
        print(f"R={R:7.1f}: response rel. to anchor per unit m: flare {resp['flare9']:+.5f}, "
              f"shadow {resp['shadow']:+.5f}, linear lapse prediction (shell outside probe) {lin:+.5f}")
