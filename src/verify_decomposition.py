"""
Numerical check of the proposed core result:

  Joint Fisher information on a deformation eps (marginalised over the common
  Kerr parameters theta) splits exactly into
     I_eps = sum_k ||(1-H_k) g_k||^2_{W_k}          (within-sector misfit)
           + ||(1-P) b||^2_{C^{-1}}                   (between-sector, from differential bias)
  where b_k = F_k^{-1} J_k^T W_k g_k is the Cutler-Vallisneri bias of sector k's Kerr fit.

Also checks: directed/orthogonal split of T^2 (Cochran), power gain, confusion angle.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(1)

def block_diag(*Ms):
    n = sum(M.shape[0] for M in Ms); m = sum(M.shape[1] for M in Ms)
    out = np.zeros((n, m)); i = j = 0
    for M in Ms:
        out[i:i+M.shape[0], j:j+M.shape[1]] = M; i += M.shape[0]; j += M.shape[1]
    return out

def random_spd(n):
    A = rng.normal(size=(n, n)); return A @ A.T + n*np.eye(n)

# ---------- 1. Pythagorean information split ----------
p = 2                      # (M, a)
n_obs = [2, 3, 4]          # observables per sector (>= p)
K = len(n_obs)
Js = [rng.normal(size=(n, p)) for n in n_obs]
Ws = [np.linalg.inv(random_spd(n)) for n in n_obs]
gs = [rng.normal(size=(n, 1)) for n in n_obs]

# joint
J = np.vstack(Js); W = block_diag(*Ws); g = np.vstack(gs)
F = J.T @ W @ J
I_joint = (g.T @ W @ g - g.T @ W @ J @ np.linalg.solve(F, J.T @ W @ g)).item()

# within
I_within = 0.0; bs = []; Sigmas = []
for Jk, Wk, gk in zip(Js, Ws, gs):
    Fk = Jk.T @ Wk @ Jk
    Hk = Jk @ np.linalg.solve(Fk, Jk.T @ Wk)
    res = gk - Hk @ gk
    I_within += (res.T @ Wk @ res).item()
    bs.append(np.linalg.solve(Fk, Jk.T @ Wk @ gk)); Sigmas.append(np.linalg.inv(Fk))

# between: differential bias in summary (M,a) space
b = np.vstack(bs); C = block_diag(*Sigmas); Ci = np.linalg.inv(C)
A = np.kron(np.ones((K, 1)), np.eye(p))
P = A @ np.linalg.solve(A.T @ Ci @ A, A.T @ Ci)
u = (np.eye(K*p) - P) @ b
I_between = (u.T @ Ci @ u).item()

print("1. Information split")
print(f"   joint      I_eps = {I_joint:.10f}")
print(f"   within + between = {I_within:.10f} + {I_between:.10f} = {I_within+I_between:.10f}")

# Common-mode bias is invisible: add same shift to all sectors
b_shift = b + np.kron(np.ones((K, 1)), rng.normal(size=(p, 1)))
u2 = (np.eye(K*p) - P) @ b_shift
print(f"   between-info after adding a common bias to all sectors: {(u2.T@Ci@u2).item():.10f} (unchanged)")

# ---------- 2. Cochran split + power by Monte Carlo ----------
print("\n2. Directed vs omnibus test (Monte Carlo, K=3, p=2, nu=4)")
L = np.linalg.cholesky(C)
nu = (K-1)*p
def stats_for(eps, N=200_000):
    theta = (L @ rng.normal(size=(K*p, N))) + eps*b       # truth theta*=0 wlog
    r = (np.eye(K*p) - P) @ theta
    T2 = np.einsum('in,ij,jn->n', r, Ci, r)
    Tpar = ((u.T @ Ci @ r)**2 / (u.T @ Ci @ u)).ravel()
    return T2, Tpar, T2 - Tpar
T2, Tpar, Tperp = stats_for(0.0)
print(f"   H0 means: T2={T2.mean():.3f} (nu={nu}), T_par={Tpar.mean():.3f} (1), T_perp={Tperp.mean():.3f} ({nu-1})")
print(f"   H0 corr(T_par, T_perp) = {np.corrcoef(Tpar, Tperp)[0,1]:+.4f}  (independent -> ~0)")
lam_target = 6.0
eps = np.sqrt(lam_target / I_between)
T2, Tpar, Tperp = stats_for(eps)
c_om = stats.chi2.ppf(0.95, nu); c_dir = stats.chi2.ppf(0.95, 1)
print(f"   H1 with noncentrality lambda={lam_target}:")
print(f"     omnibus  power  MC={np.mean(T2>c_om):.3f}  theory={stats.ncx2.sf(c_om, nu, lam_target):.3f}")
print(f"     directed power  MC={np.mean(Tpar>c_dir):.3f}  theory={stats.ncx2.sf(c_dir, 1, lam_target):.3f}")
print(f"     T_perp mean under H1 = {Tperp.mean():.3f} (stays {nu-1}: deformation leaves no trace there)")

# ---------- 3. Confusion angle ----------
def confusion(bvec, Cmat, Kk, pp, sector, direction):
    Cinv = np.linalg.inv(Cmat)
    AA = np.kron(np.ones((Kk, 1)), np.eye(pp))
    PP = AA @ np.linalg.solve(AA.T @ Cinv @ AA, AA.T @ Cinv)
    uu = (np.eye(Kk*pp) - PP) @ bvec
    e = np.zeros((Kk*pp, 1)); e[sector*pp:(sector+1)*pp] = np.reshape(direction, (pp, 1))
    s = (np.eye(Kk*pp) - PP) @ e
    return (uu.T@Cinv@s).item()/np.sqrt((uu.T@Cinv@uu).item()*(s.T@Cinv@s).item())

print("\n3. Confusion angle |cos phi| between a deformation and a single-sector systematic")
for trial in range(3):
    bb = rng.normal(size=(2, 1)); CC = np.diag(rng.uniform(0.5, 2, 2))
    print(f"   K=2, p=1 (nu=1): |cos| = {abs(confusion(bb, CC, 2, 1, 1, [1.0])):.6f}")
bb = rng.normal(size=(3, 1)); CC = np.diag(rng.uniform(0.5, 2, 3))
print(f"   K=3, p=1 (nu=2): |cos| = {abs(confusion(bb, CC, 3, 1, 2, [1.0])):.4f}  (<1: distinguishable in principle)")
