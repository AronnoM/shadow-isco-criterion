"""
3rd-order WKB (Iyer & Will 1987) quasinormal frequencies for test fields of spin s
in a static metric  ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2,
   V(r) = f [ l(l+1)/r^2 + (1 - s^2) f'/r ]      (s = 0 scalar, s = 1 EM;
                                                  s = 2 reproduces Regge-Wheeler only for Schwarzschild)
Units M = 1. Deformation: f = 1 - 2/r + d (1/r)^n.
"""
import numpy as np
import sympy as sp
from functools import lru_cache
from scipy.optimize import brentq

r_, d_ = sp.symbols('r d', real=True)


@lru_cache(maxsize=None)
def _derivs(n, l, s):
    f = 1 - 2 / r_ + d_ * r_ ** (-n)
    V = f * (l * (l + 1) / r_ ** 2 + (1 - s ** 2) * sp.diff(f, r_) / r_)
    dV_dr = sp.diff(V, r_)
    Ds = [V]
    cur = V
    for k in range(6):
        cur = sp.simplify(f * sp.diff(cur, r_)) if k < 2 else f * sp.diff(cur, r_)
        Ds.append(cur)
    fn = [sp.lambdify((r_, d_), D, 'numpy') for D in Ds]
    dVr = sp.lambdify((r_, d_), dV_dr, 'numpy')
    return fn, dVr


def qnm_wkb3(d, n=2, l=2, s=0, overtone=0):
    fn, dVr = _derivs(n, l, s)
    r0 = brentq(lambda rr: dVr(rr, d), 2.3, 4.5)
    V0, V1, V2, V3, V4, V5, V6 = [g(r0, d) for g in fn]
    a = overtone + 0.5
    Lam = (1 / np.sqrt(-2 * V2 + 0j)) * ((1 / 8) * (V4 / V2) * (0.25 + a ** 2)
                                        - (1 / 288) * (V3 / V2) ** 2 * (7 + 60 * a ** 2))
    Om = (1 / (-2 * V2)) * ((5 / 6912) * (V3 / V2) ** 4 * (77 + 188 * a ** 2)
                            - (1 / 384) * (V3 ** 2 * V4 / V2 ** 3) * (51 + 100 * a ** 2)
                            + (1 / 2304) * (V4 / V2) ** 2 * (67 + 68 * a ** 2)
                            + (1 / 288) * (V3 * V5 / V2 ** 2) * (19 + 28 * a ** 2)
                            - (1 / 288) * (V6 / V2) * (5 + 4 * a ** 2))
    w2 = (V0 + np.sqrt(-2 * V2 + 0j) * Lam) - 1j * a * np.sqrt(-2 * V2 + 0j) * (1 + Om)
    w = np.sqrt(w2)
    if w.real < 0:
        w = -w
    return w


if __name__ == "__main__":
    for s, l, ref in [(2, 2, 0.37367 - 0.08896j), (0, 2, 0.48364 - 0.09677j), (1, 2, 0.45759 - 0.09501j)]:
        w = qnm_wkb3(0.0, n=2, l=l, s=s)
        print(f"Schwarzschild s={s} l={l}: WKB3 {w.real:.5f} {w.imag:+.5f}i   (exact {ref.real:.5f} {ref.imag:+.5f}i)")
