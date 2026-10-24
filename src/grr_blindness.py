"""
Lyapunov exponent of the light ring and its deformation kernels (corrected sign convention).

Metric: ds^2 = -A dt^2 + B dr^2 + r^2 dOmega^2.  Null geodesics: dt/dlam = E/A, dphi/dlam = L/r^2,
    (dr/dlam)^2 = R(r) = (E^2/A - L^2/r^2) / B.
At the light ring R = R' = 0 and R'' > 0 (unstable orbit), so the coordinate-time Lyapunov exponent
    lambda = sqrt( R'' / (2 (dt/dlam)^2) )
is real and positive. Check: Schwarzschild lambda = 1/(3 sqrt3 M).
Kernels: mass inferred from a rate scales as M_inf ∝ 1/lambda  ->  beta = -d ln(lambda)/d d.
"""
import json
import sympy as sp

r, d = sp.symbols('r d', real=True)
E, L = sp.symbols('E L', positive=True)


def lyapunov(A, B):
    # light ring from d/dr (A/r^2) = 0 (depends on A only); solve perturbatively around r = 3
    rc0 = sp.nsolve(sp.diff(A / r ** 2, r).subs(d, 0), r, 3)
    y = sp.symbols('y')
    eq = sp.series(sp.diff(A / r ** 2, r).subs(r, rc0 + d * y), d, 0, 2).removeO()
    ysol = sp.solve(sp.Eq(eq.coeff(d, 1), 0), y)
    rc = rc0 + d * (ysol[0] if ysol else 0)
    b2 = rc ** 2 / A.subs(r, rc)                       # (L/E)^2 at the light ring
    R = (E ** 2 / A - L ** 2 / r ** 2) / B
    Rpp = sp.diff(R, r, 2).subs(r, rc).subs(L, E * sp.sqrt(b2))
    tdot2 = (E / A.subs(r, rc)) ** 2
    return sp.sqrt(sp.simplify(Rpp / (2 * tdot2)))


out = {}
A_s = 1 - 2 / r
lam0 = sp.nsimplify(sp.simplify(lyapunov(A_s, 1 / A_s).subs(d, 0)))
print(f"Schwarzschild lambda = {lam0}  (= {float(lam0):.6f}; expected 1/(3 sqrt 3) = {1/(3*3**0.5):.6f})")
assert lam0.is_real and float(lam0) > 0 and abs(float(lam0) - 1 / (3 * 3 ** 0.5)) < 1e-12
out['schwarzschild_lambda'] = float(lam0)

for n in [2, 3, 4]:
    # (a) pure g_rr deformation: A unchanged, B -> (1 + d r^-n)/A
    lam = lyapunov(A_s, (1 + d * r ** (-n)) / A_s)
    k_rr = sp.nsimplify(sp.simplify(-sp.diff(sp.log(lam), d).subs(d, 0)))
    # (b) g_tt deformation with g_rr = 1/A (the family used throughout)
    A_t = 1 - 2 / r + d * r ** (-n)
    lam_t = lyapunov(A_t, 1 / A_t)
    k_tt = sp.nsimplify(sp.simplify(-sp.diff(sp.log(lam_t), d).subs(d, 0)))
    print(f"n={n}: omega_I mass kernel  g_rr-deformation {k_rr} (expected +1/2*3^-n = {sp.Rational(1, 2) * 3**(-n)}),"
          f"  g_tt-deformation {k_tt}")
    assert sp.simplify(k_rr - sp.Rational(1, 2) * sp.Integer(3) ** (-n)) == 0
    out[n] = dict(grr_kernel=str(k_rr), gtt_kernel=str(k_tt))
print("Orbit, ISCO and shadow kernels for pure g_rr deformations are 0: they depend on A only.")
json.dump(out, open('results/grr_blindness.json', 'w'), indent=1)
