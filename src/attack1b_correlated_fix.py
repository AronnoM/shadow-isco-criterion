"""
Attack 1(b) follow-up: with cross-sector noise correlations, does the split survive if
sector summaries are computed by a JOINT generalised-least-squares fit with separate
Kerr parameters per sector (model S1 fitted with the full noise matrix), instead of
each sector fitting its own data alone?
"""
import json, numpy as np
from core import info_split, block_diag, wproj, residual_projector
rng = np.random.default_rng(11)
def spd(n, cond=5.0):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n))); ev = np.exp(rng.uniform(0, np.log(cond), n)); return Q @ np.diag(ev) @ Q.T
p = 2; sizes = [3, 3, 4]
res = []
for c in [0.1, 0.3, 0.5]:
    errs = []
    for t in range(300):
        Js = [rng.normal(size=(n, p)) for n in sizes]; Gs = [rng.normal(size=(n, 1)) for n in sizes]
        Ns = [spd(n) for n in sizes]
        D = block_diag(*Ns); L = np.linalg.cholesky(D); n = D.shape[0]
        Z = rng.normal(size=(n, n)); X = Z @ Z.T / n; Xs = X / np.sqrt(np.outer(np.diag(X), np.diag(X)))
        mask = 1 - block_diag(*[np.ones((k, k)) for k in sizes]); M = np.eye(n) + c * Xs * mask
        if np.linalg.eigvalsh(M).min() < 0.05: continue
        Nf = L @ M @ L.T; W = np.linalg.inv(Nf)
        s = info_split(Js, Gs, Ns, Nfull=Nf)
        # joint GLS with separate parameters per sector: theta_sep = (X1^T W X1)^{-1} X1^T W y
        X1 = block_diag(*Js); F1 = X1.T @ W @ X1; L1 = np.linalg.solve(F1, X1.T @ W)
        C = np.linalg.inv(F1)                      # full covariance of the jointly-fitted summaries
        b = L1 @ np.vstack(Gs)                     # their bias
        A = np.kron(np.ones((len(sizes), 1)), np.eye(p))
        R, Ci = residual_projector(A, C); U = R @ b
        Ib = (U.T @ Ci @ U).item()
        errs.append(abs(Ib - s['between_abstract'].item()) / s['total'].item())
    res.append(dict(c=c, max_rel_err=float(np.max(errs)), n=len(errs)))
    print(f"c={c}: joint-GLS summaries reproduce the abstract between-term to max rel. err {np.max(errs):.1e} ({len(errs)} draws)")
json.dump(res, open('results/attack1b.json', 'w'), indent=1)
