"""
Stage 2: the environmental theorem, done properly.

 (1) Conditional equal-response lemma (Kantorovich bound) -- only valid when the environment's
     response is the same at the two strong-field probes.
 (2) Thin-shell counterexample: response kernels with lapse normalisation and ray-traced astrometry.
 (3) General bound: environments are non-negative combinations of shells -> a convex cone of
     templates; the worst-case degeneracy is the angle between the deformation template and that cone.
 (4) Stellar-orbit priors: S2 apsidal precession from shells and from GRAVITY's Plummer bound.
Units G = c = M = 1; Sgr A*: 1 AU = 23.56 M (M = 4.30e6 Msun).
"""
import json
import warnings
import numpy as np
from scipy.optimize import nnls
from scipy.integrate import quad
from core import kernel, residual_projector, whiten
from thin_shell import orbit_mass, shadow_mass

warnings.filterwarnings('ignore')
out = {}
AU = 23.56
r_peri, ecc = 120 * AU, 0.88
r_apo = r_peri * (1 + ecc) / (1 - ecc)
R_FLARE, R_ANCHOR = 9.0, r_peri

# ------------------------------------------------------------ (2) response kernels per unit shell mass
Rgrid = np.unique(np.concatenate([np.linspace(3.2, 8.99, 40), np.linspace(9.01, 30, 50),
                                  np.logspace(np.log10(30), np.log10(R_ANCHOR * 0.999), 60)]))
dm = 1e-5
resp = []
for R in Rgrid:
    a_resp = 1.0                                           # anchor: shell inside S2 pericentre -> sees M + m
    f_resp = (orbit_mass(R_FLARE, dm, R) - 1) / dm
    s_resp = (shadow_mass(dm, R) - 1) / dm
    resp.append((R, a_resp, f_resp, s_resp))
resp = np.array(resp)
out['response_grid'] = resp.tolist()
print("(2) thin-shell responses per unit m (absolute dM_inf/m): R, anchor, flare(9M), shadow")
for R in [4, 6, 8.5, 9.5, 12, 30, 100, 1000]:
    i = np.argmin(abs(resp[:, 0] - R))
    print(f"    R={resp[i,0]:8.2f}: {resp[i,1]:+.4f} {resp[i,2]:+.4f} {resp[i,3]:+.4f}")

# ------------------------------------------------------------ (1)+(3) cone analysis
def templates(n):
    u = np.array([kernel(n, 'orbit', R_ANCHOR)] * 2 + [kernel(n, 'orbit', R_FLARE), kernel(n, 'shadow')])
    S = np.column_stack([[r_[1], r_[1], r_[2], r_[3]] for r_ in resp])   # GRAVITY, Keck, flare, EHT
    return u, S


def cone_cos(u, S, C, mask):
    """max cos between the projected deformation template and the cone spanned by S[:, mask]."""
    A = np.ones((len(u), 1))
    Rp, Ci = residual_projector(A, C)
    Wh = whiten(Ci)
    U = Wh @ (Rp @ u); Sm = Wh @ (Rp @ S[:, mask])
    best = 0.0; sol = None
    for sign in (+1, -1):                                  # deformation of either sign
        c, res = nnls(Sm, sign * U)
        cs = np.sqrt(max(0.0, 1 - res ** 2 / (U @ U)))
        if cs > best:
            best, sol = cs, (sign, c)
    return best, sol


C_today = np.diag([0.0043, 0.0064, 0.40, 0.20]) ** 2
C_anchor = np.diag([1e-5, 1e-5, 0.40, 0.20]) ** 2         # precise-anchor limit
res3 = {}
for n in [2, 3]:
    u, S = templates(n)
    rows = []
    for Rmin in [3.2, 6.0, 9.0, 9.5, 12, 20, 50, 200]:
        mask = resp[:, 0] >= Rmin
        c1, _ = cone_cos(u, S, C_today, mask); c2, _ = cone_cos(u, S, C_anchor, mask)
        rows.append(dict(Rmin=Rmin, cos_today=c1, cos_precise_anchor=c2))
    res3[n] = rows
    print(f"\n(3) n={n}: max |cos| between d_n and admissible environments with shells at R >= Rmin")
    for row in rows:
        print(f"    Rmin={row['Rmin']:6.1f} M: today {row['cos_today']:.4f}   precise anchor {row['cos_precise_anchor']:.4f}")
    k = abs(kernel(n, 'orbit', R_FLARE) / kernel(n, 'shadow'))
    print(f"    equal-response lemma value 2sqrt(k)/(1+k) = {2*np.sqrt(k)/(1+k):.4f} (k = {k:.3f})")
out['cone'] = res3

# explicit mimicking environment for d2 = 0.5 using stable shells (R >= 6M): two-shell solution
u, S = templates(2)
d2 = 0.5
target = d2 * (u - u[0])                                   # relative to the anchor (anchor kernel ~ 0)
# far shell: pick R_far = 1000 M; inner shell: R_in in [6, 9)
i_far = np.argmin(abs(resp[:, 0] - 1000)); i_in = np.argmin(abs(resp[:, 0] - 6.0))
rel_far = resp[i_far, 2:] - resp[i_far, 1]; rel_in = resp[i_in, 2:] - resp[i_in, 1]
coef = np.linalg.solve(np.column_stack([rel_far, rel_in]), target[2:])
out['mimic_d2_0.5'] = dict(R_far=resp[i_far, 0], m_far=coef[0], R_in=resp[i_in, 0], m_in=coef[1])
print(f"\n    exact mimic of d2 = 0.5 relative to S2: shell {coef[0]*100:.2f}% M at R = {resp[i_far,0]:.0f} M"
      f" + shell {coef[1]*100:.2f}% M at R = {resp[i_in,0]:.1f} M  (masses in units of M)")


# ------------------------------------------------------------ (4) S2 precession
def precession(Phi):
    """Apsidal advance per orbit (rad) for fixed peri/apo in total potential -1/r + Phi(r)."""
    Pt = lambda r: -1.0 / r + Phi(r)
    L2 = 2 * (Pt(r_apo) - Pt(r_peri)) / (1 / r_peri ** 2 - 1 / r_apo ** 2)
    E = Pt(r_peri) + L2 / (2 * r_peri ** 2)
    rm, Dl = (r_apo + r_peri) / 2, (r_apo - r_peri) / 2
    def integrand(chi):
        r = rm - Dl * np.cos(chi)
        Q = 2 * (E - Pt(r)) - L2 / r ** 2
        return np.sqrt(L2) / r ** 2 * Dl * np.sin(chi) / np.sqrt(max(Q, 1e-300))
    val, _ = quad(integrand, 0, np.pi, limit=400, epsabs=1e-14, epsrel=1e-12, points=None)
    return 2 * val - 2 * np.pi


SP = 6 * np.pi / (((r_peri + r_apo) / 2) * (1 - ecc ** 2))
a_pl = 0.3 * 8277 * AU                                       # 0.3 arcsec at R0 = 8277 pc, in M
dw_pl = precession(lambda r: -1e-3 / np.sqrt(r ** 2 + a_pl ** 2)) / 1e-3   # per unit f_Pl
f_lim = 2 * 2.4e-3                                           # 2-sigma width of GRAVITY's 4-star f_Pl
dw_max = abs(dw_pl) * f_lim
print(f"\n(4) S2: Schwarzschild precession {SP*180/np.pi*60:.1f} arcmin/orbit; Plummer (a = 0.3\") "
      f"{dw_pl*180/np.pi*60:.2f} arcmin per unit f_Pl; allowed |extra| <= {dw_max*180/np.pi*60:.2f} arcmin "
      f"(f_Pl < {f_lim:.1e})")
shell_lim = []
for R in np.logspace(np.log10(r_peri * 0.5), np.log10(r_apo * 1.5), 60):
    def Phi(r, R=R, m=1e-4):
        return -m / max(r, R)
    dw = precession(Phi) / 1e-4
    m_max = dw_max / abs(dw) if abs(dw) > 1e-12 else np.inf
    shell_lim.append((R, dw, m_max))
shell_lim = np.array(shell_lim)
out['s2_shell_limits'] = shell_lim.tolist()
out['s2'] = dict(r_peri=r_peri, r_apo=r_apo, SP_arcmin=SP * 180 / np.pi * 60, f_lim=f_lim,
                 dw_max_arcmin=dw_max * 180 / np.pi * 60)
inside = shell_lim[shell_lim[:, 0] < r_peri]
within = shell_lim[(shell_lim[:, 0] > r_peri) & (shell_lim[:, 0] < r_apo)]
print(f"    shells inside pericentre (R < {r_peri:.0f} M): max allowed mass = "
      f"{'unbounded' if np.all(np.isinf(inside[:,2])) else inside[:,2].max()}")
print(f"    shells across the orbit: tightest bound {within[:,2].min()*100:.3f}% M at R = {within[np.argmin(within[:,2]),0]:.0f} M;"
      f" loosest {within[:,2].max()*100:.2f}% M")
json.dump(out, open('results/env_theorem.json', 'w'), indent=1, default=float)
