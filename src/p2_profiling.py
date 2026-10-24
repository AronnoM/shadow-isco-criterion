"""
P2 with nuisance profiling: when does compressing each sector to a summary preserve the
between-sector information?

Claim (proved in the notes, checked here):
  (i)  If sectors are independent and each reports the profiled GLS estimate of its identifiable
       theta-combinations (private nuisances profiled), then I_b(summary) = I_b(observable).
  (ii) A linear compression z_k = L_k y_k preserves I_b for every deformation direction iff the
       rows of Jt_k^T W_k lie in the row space of L_k (Jt_k = profiled Jacobian).
       Dropping an identifiable combination therefore loses information for generic G.
  (iii) Ignoring a private nuisance (fixing it) changes the null hypothesis; a nuisance shared by
       several sectors must be promoted to a common parameter, otherwise I_b refers to a weaker null.
"""
import json
import numpy as np
from core import info_split, summary_from_sectors, between_info_summary, block_diag, wproj

rng = np.random.default_rng(21)
out = {}


def spd(n):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q @ np.diag(np.exp(rng.uniform(0, np.log(5), n))) @ Q.T


p = 2
errs = []
for t in range(300):
    spec = [(5, 2, 1), (4, 2, 2), (3, 1, 1)]          # (n_obs, rank J, n private nuisances)
    Js, Gs, Ns, Bs = [], [], [], []
    for n, rk, nb in spec:
        Js.append(rng.normal(size=(n, rk)) @ rng.normal(size=(rk, p)))
        Gs.append(rng.normal(size=(n, 1))); Ns.append(spd(n)); Bs.append(rng.normal(size=(n, nb)))
    s = info_split(Js, Gs, Ns, Bs=Bs)
    sm = summary_from_sectors(Js, Gs, Ns, Bs=Bs)
    Ib = between_info_summary(sm['A'], sm['b'], block_diag(*sm['Sigmas'])).item()
    errs.append(abs(Ib - s['between_abstract'].item()) / max(s['total'].item(), 1e-300))
out['i_profiled_max_rel_err'] = float(np.max(errs))
print(f"(i)  profiled summaries reproduce I_b: max rel. error {np.max(errs):.1e} over 300 draws "
      f"(ranks after profiling {sm['ranks']})")

# (ii) compression that drops one identifiable combination of one sector
loss = []
for t in range(300):
    Js = [rng.normal(size=(4, p)) for _ in range(3)]
    Gs = [rng.normal(size=(4, 1)) for _ in range(3)]
    Ns = [spd(4) for _ in range(3)]
    sm = summary_from_sectors(Js, Gs, Ns)
    C = block_diag(*sm['Sigmas'])
    Ib_full = between_info_summary(sm['A'], sm['b'], C).item()
    keep = [0, 1, 2, 3, 4, 5]; keep.remove(1)            # sector 0 reports only its first combination
    A2 = sm['A'][keep]; b2 = sm['b'][keep]; C2 = C[np.ix_(keep, keep)]
    Ib_comp = between_info_summary(A2, b2, C2).item()
    loss.append(1 - Ib_comp / Ib_full)
loss = np.array(loss)
out['ii_drop_component_loss'] = dict(min=float(loss.min()), median=float(np.median(loss)), max=float(loss.max()))
print(f"(ii) dropping one combination of one sector: information lost min {loss.min():.3f}, "
      f"median {np.median(loss):.3f}, max {loss.max():.3f} (never negative)")
# a basis change of the summaries (rows of L_k mixed by an invertible matrix) loses nothing
Js = [rng.normal(size=(4, p)) for _ in range(3)]; Gs = [rng.normal(size=(4, 1)) for _ in range(3)]
Ns = [spd(4) for _ in range(3)]
sm = summary_from_sectors(Js, Gs, Ns); C = block_diag(*sm['Sigmas'])
Mmix = block_diag(*[rng.normal(size=(2, 2)) for _ in range(3)])
Ib_a = between_info_summary(sm['A'], sm['b'], C).item()
Ib_b = between_info_summary(Mmix @ sm['A'], Mmix @ sm['b'], Mmix @ C @ Mmix.T).item()
out['ii_basis_change_rel_diff'] = abs(Ib_a - Ib_b) / Ib_a
print(f"     invertible mixing of each sector's summaries: rel. change {abs(Ib_a-Ib_b)/Ib_a:.1e}")

# (iii) a nuisance shared by all sectors (e.g. distance) -- three treatments
Js = [rng.normal(size=(4, p)) for _ in range(3)]; Gs = [rng.normal(size=(4, 1)) for _ in range(3)]
Ns = [spd(4) for _ in range(3)]; Ds = [rng.normal(size=(4, 1)) for _ in range(3)]
# correct: promote the shared nuisance to a common parameter
s_common = info_split([np.hstack([J, D]) for J, D in zip(Js, Ds)], Gs, Ns)
# weaker null: give each sector its own copy
s_private = info_split(Js, Gs, Ns, Bs=Ds)
# wrong: ignore it altogether (fixed at its true value) -- the information is that of a different model
s_fixed = info_split(Js, Gs, Ns)
out['iii_shared_nuisance'] = dict(I_b_common=s_common['between_abstract'].item(),
                                   I_b_private_copies=s_private['between_abstract'].item(),
                                   I_b_fixed=s_fixed['between_abstract'].item(),
                                   nu_common=s_common['nu'], nu_private=s_private['nu'], nu_fixed=s_fixed['nu'])
print(f"(iii) shared nuisance: I_b common-parameter {s_common['between_abstract'].item():.3f} (nu={s_common['nu']}),"
      f" private copies {s_private['between_abstract'].item():.3f} (nu={s_private['nu']}),"
      f" fixed {s_fixed['between_abstract'].item():.3f} (nu={s_fixed['nu']})")
json.dump(out, open('results/p2_profiling.json', 'w'), indent=1, default=float)
