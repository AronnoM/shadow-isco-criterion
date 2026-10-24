"""
Direct numerical QNMs (Chebyshev pseudo-spectral, ingoing Eddington-Finkelstein + compactification)
for test fields of spin s in  ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2,  f = 1 - 2/r + d r^-n  (M = 1).

Radial equation:  d^2R/dr*^2 + (w^2 - V) R = 0,  V = f [ l(l+1)/r^2 + (1 - s^2) f'/r ].
Ansatz R = exp(-i w r*) exp(2 i w r) r^(4 i w) psi(r): ingoing at the horizon, outgoing at infinity,
psi regular at both ends. In u = r_h / r in [0, 1] the equation is quadratic in w; solved as a
generalised eigenproblem and filtered by agreement between two resolutions.
"""
import numpy as np
import sympy as sp
from functools import lru_cache
from scipy.linalg import eig
from scipy.optimize import brentq

u, w, d, rh = sp.symbols('u w d r_h')
psi = sp.Function('psi')


@lru_cache(maxsize=None)
def _coeffs(n, l, s):
    r = rh / u
    f = 1 - 2 / r + d * r ** (-n)
    # d/dr = -(u^2/rh) d/du ; build the operator acting on phi(r) = e^{2iwr} r^{4iw} psi(u)
    Dr = lambda expr: -(u ** 2 / rh) * sp.diff(expr, u)
    fp = Dr(f)
    V = f * (l * (l + 1) / r ** 2 + (1 - s ** 2) * fp / r)
    pre = sp.exp(2 * sp.I * w * r) * r ** (4 * sp.I * w)
    phi = pre * psi(u)
    expr = f * Dr(f * Dr(phi)) - 2 * sp.I * w * f * Dr(phi) - V * phi
    # divide by f (single zero at the horizon) and by u^2 (common factor at infinity), so both
    # boundaries are regular singular points and collocation imposes regularity automatically
    expr = sp.expand(sp.simplify(expr / (pre * f * u ** 2)))
    P2, P1, P0 = [sp.Wild(k) for k in 'abc']
    c2 = sp.simplify(expr.coeff(sp.Derivative(psi(u), (u, 2))))
    rest = sp.expand(expr - c2 * sp.Derivative(psi(u), (u, 2)))
    c1 = sp.simplify(rest.coeff(sp.Derivative(psi(u), u)))
    c0 = sp.simplify(sp.expand(rest - c1 * sp.Derivative(psi(u), u)).coeff(psi(u)))
    out = []
    for c in (c2, c1, c0):
        poly = sp.Poly(sp.expand(c), w)
        terms = {k[0]: sp.lambdify((u, d, rh), v, 'numpy') for k, v in zip(poly.monoms(), poly.coeffs())}
        out.append(terms)
    return out


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def horizon(dval, n):
    return brentq(lambda r: 1 - 2 / r + dval * r ** (-n), 1.0 + 1e-9, 2.5) if dval > 0 else \
        brentq(lambda r: 1 - 2 / r + dval * r ** (-n), 1.5, 4.0)


def spectrum(dval, n, l, s, N):
    rh_v = horizon(dval, n)
    D, x = cheb(N)
    uu = (x + 1) / 2                                  # u in [0, 1]
    D1 = 2 * D; D2 = D1 @ D1
    coeffs = _coeffs(n, l, s)
    mats = []
    for power in (0, 1, 2):
        M = np.zeros((N + 1, N + 1), complex)
        for c, Dk in zip(coeffs, (D2, D1, np.eye(N + 1))):
            if power in c:
                vals = np.asarray(c[power](uu, dval, rh_v), complex) * np.ones_like(uu)
                M += np.diag(vals) @ Dk
        mats.append(M)
    M0, M1, M2 = mats
    Z = np.zeros_like(M0); I = np.eye(N + 1)
    Aa = np.block([[Z, I], [-M0, -M1]]); Bb = np.block([[I, Z], [Z, M2]])
    ev = eig(Aa, Bb, right=False)
    ev = ev[np.isfinite(ev)]
    return ev


def fundamental(dval, n, l, s, N=60, guess=None):
    """Fundamental mode (Re w > 0, smallest |Im w|) confirmed at two resolutions."""
    e1, e2 = spectrum(dval, n, l, s, N), spectrum(dval, n, l, s, N + 16)
    cand = [z for z in e1 if z.real > 0.05 and -1.0 < z.imag < -0.01]
    good = []
    for z in cand:
        dz = np.min(np.abs(e2 - z))
        if dz < 1e-8 * abs(z) + 1e-12:
            good.append((z, dz))
    if not good:
        raise RuntimeError("no converged mode")
    if guess is not None:
        z, dz = min(good, key=lambda t: abs(t[0] - guess))
    else:
        z, dz = max(good, key=lambda t: t[0].imag)
    return z, dz


if __name__ == "__main__":
    for s, l, ref in [(2, 2, 0.373672 - 0.088962j), (0, 2, 0.483644 - 0.096759j), (1, 2, 0.457596 - 0.095004j)]:
        z, dz = fundamental(0.0, 2, l, s)
        print(f"Schwarzschild s={s} l={l}: {z.real:.8f} {z.imag:+.8f}i  (ref {ref.real:.6f} {ref.imag:+.6f}i), "
              f"resolution change {dz:.1e}")
