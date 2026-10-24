"""
Fundamental-bias kernels for probes of a deformed static black hole.

Metric: ds^2 = -A dt^2 + dr^2/A + r^2 dOmega^2,  A(r) = 1 - 2M/r + d_n (M/r)^n.
Each probe infers a mass M_inf assuming Schwarzschild. Bias kernel
    beta_k(n) = (1/M) dM_inf/d d_n |_{d_n=0}.
n = 1 is a pure mass renormalisation (must give the same kernel, -1/2, for every probe).
"""
import sympy as sp
import numpy as np

r, M, d = sp.symbols('r M d', positive=True)

def kernels(n):
    A = 1 - 2*M/r + d*(M/r)**n
    Ap, App = sp.diff(A, r), sp.diff(A, r, 2)
    out = {}

    # (a) circular orbit with r and Omega both measured (astrometric orbit / hotspot):
    #     Schwarzschild: Omega^2 = M/r^3 exactly  ->  M_inf = r^3 Omega^2 = r^2 A'/2
    Minf = r**2*Ap/2
    out['orbit(r)'] = sp.simplify(sp.diff(Minf, d).subs(d, 0)/M)

    # (b) ISCO orbital frequency alone (e.g. QPO at ISCO): Schw Omega_isco = 1/(6 sqrt6 M)
    L2 = r**3*Ap/(2*A - r*Ap)
    cond = sp.diff(L2, r)
    # perturb: r_isco = 6M + d*x
    x = sp.symbols('x')
    cond_lin = sp.series(sp.simplify(cond.subs(r, 6*M + d*x*M)), d, 0, 2).removeO()
    x_sol = sp.solve(sp.Eq(sp.simplify(cond_lin.coeff(d, 1)), 0), x)[0]
    Om = sp.sqrt(Ap/(2*r))
    Om_isco = Om.subs(r, 6*M + d*x_sol*M)
    dlnOm = sp.diff(sp.log(Om_isco), d).subs(d, 0)
    out['ISCO freq'] = sp.nsimplify(sp.simplify(-dlnOm))      # frequency ~ 1/M

    # (c) shadow / critical impact parameter b_c = r_c/sqrt(A(r_c)), r_c extremises A/r^2
    #     envelope theorem: d(b_c^{-2}) = dA(r_c)/r_c^2 at r_c = 3M
    bc_m2 = (A/r**2)
    dln_bc = -sp.Rational(1, 2)*(sp.diff(bc_m2, d)/bc_m2).subs({d: 0, r: 3*M})
    out['shadow'] = sp.simplify(dln_bc)                         # length ~ M

    # (d) eikonal QNM real part  omega_R = l * Omega_c = l / b_c  -> identical to shadow
    out['QNM omega_R (eikonal)'] = out['shadow']

    # (e) eikonal QNM damping: Lyapunov exponent at light ring (Cardoso et al. 2009)
    #     lambda^2 = A (2A - r^2 A'') / (2 r^2) at r_c ; need r_c perturbed
    y = sp.symbols('y')
    rc_cond = sp.diff(A/r**2, r)
    rc_lin = sp.series(sp.simplify(rc_cond.subs(r, 3*M + d*y*M)), d, 0, 2).removeO()
    y_sol = sp.solve(sp.Eq(sp.simplify(rc_lin.coeff(d, 1)), 0), y)[0]
    lam2 = (A*(2*A - r**2*App)/(2*r**2)).subs(r, 3*M + d*y_sol*M)
    dln_lam = sp.Rational(1, 2)*sp.diff(sp.log(lam2), d).subs(d, 0)
    out['QNM omega_I (eikonal)'] = sp.simplify(-dln_lam)        # rate ~ 1/M
    return out

rows = {}
for n in [1, 2, 3, 4]:
    rows[n] = kernels(n)

probes = ['orbit(r)', 'ISCO freq', 'shadow', 'QNM omega_R (eikonal)', 'QNM omega_I (eikonal)']
print("Bias kernels beta(n) = dM_inf/(M dd_n):\n")
for pr in probes:
    line = f"{pr:24s}"
    for n in rows:
        val = rows[n][pr]
        line += f"  n={n}: {str(sp.simplify(val)):>22s}"
    print(line)

print("\nNumerical values (orbit evaluated at r = 8M and r = 2800M):")
for n in rows:
    vals = {pr: float(rows[n][pr].subs(r, 8*M)) if pr == 'orbit(r)' else float(rows[n][pr]) for pr in probes}
    far = float(rows[n]['orbit(r)'].subs(r, 2800*M))
    print(f" n={n}: orbit@2800M={far:+.2e}  orbit@8M={vals['orbit(r)']:+.4f}  ISCO={vals['ISCO freq']:+.4f}  "
          f"shadow={vals['shadow']:+.4f}  wR={vals['QNM omega_R (eikonal)']:+.4f}  wI={vals['QNM omega_I (eikonal)']:+.4f}")

# ---- brute-force numerical check of the shadow and ISCO kernels for n=2 ----
from scipy.optimize import brentq, minimize_scalar
def shadow_bc(dd, n, Mv=1.0):
    A = lambda rr: 1 - 2*Mv/rr + dd*(Mv/rr)**n
    res = minimize_scalar(lambda rr: -A(rr)/rr**2, bounds=(2.2, 5), method='bounded')
    return 1/np.sqrt(A(res.x)/res.x**2)
def isco_omega(dd, n, Mv=1.0):
    A = lambda rr: 1 - 2*Mv/rr + dd*(Mv/rr)**n
    Ap = lambda rr: 2*Mv/rr**2 - n*dd*Mv**n/rr**(n+1)
    L2 = lambda rr: rr**3*Ap(rr)/(2*A(rr) - rr*Ap(rr))
    h = 1e-6
    dL2 = lambda rr: (L2(rr+h) - L2(rr-h))/(2*h)
    ri = brentq(dL2, 4.5, 8)
    return np.sqrt(Ap(ri)/(2*ri))
eps = 1e-5
for n in [2, 3]:
    k_sh = (np.log(shadow_bc(eps, n)) - np.log(shadow_bc(-eps, n)))/(2*eps)
    k_is = -(np.log(isco_omega(eps, n)) - np.log(isco_omega(-eps, n)))/(2*eps)
    print(f" finite-difference check n={n}: shadow {k_sh:+.5f} (sympy {float(rows[n]['shadow']):+.5f}),"
          f" ISCO {k_is:+.5f} (sympy {float(rows[n]['ISCO freq']):+.5f})")
