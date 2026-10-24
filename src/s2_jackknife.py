"""Jackknife of the within-sector delta_2 (precession) constraint of the Keck refit, independent errors, linearised."""
import json, numpy as np, sys
sys.path.insert(0, '.')
import s2_refit as S
d = json.load(open('results/s2_deform.json'))
th = np.array(d['independent']['theta']); err = np.array(d['independent']['err'])
S.set_noise(None)
r0 = S.residuals(th)
J = S.jac_central(lambda p: S.residuals(p), th, 0.1 * err)
h = 0.05; g = (S.residuals(th, d2=h) - S.residuals(th, d2=-h)) / (2 * h)
Jd = np.column_stack([J, g])
na, nr = len(S.t_ast), len(S.t_rv)
ta = np.concatenate([S.t_ast, S.t_ast]); tr = S.t_rv
def est(mask):
    st = -np.linalg.lstsq(Jd[mask], r0[mask], rcond=None)[0]
    F = Jd[mask].T @ Jd[mask]; cov = np.linalg.inv(F)
    return st[-1], np.sqrt(cov[-1, -1])
allm = np.ones(2 * na + nr, bool)
tests = {'all': allm}
m = allm.copy(); m[:2 * na] &= ~(ta < 2005.0); tests['drop speckle astrometry (<2005)'] = m
m = allm.copy(); m[:2 * na] &= ~(ta > 2018.0); tests['drop 2018 astrometry'] = m
m = allm.copy(); m[2 * na:] &= ~(tr > 2018.0); tests['drop 2018 RVs'] = m
m = allm.copy(); m[2 * na:] &= ~np.isin(S.src_rv, ['VLT']); tests['drop VLT RVs'] = m
for k, mm in tests.items():
    e, s = est(mm); print(f"{k:34s}: delta_2 = {e:+7.2f} +- {s:5.2f}   (f_SP-equiv {1-e/6:5.2f})")
