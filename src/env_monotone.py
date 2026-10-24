"""
Which probe sets can separate a deformation from any non-negative environment?
Shell responses are step-like: a shell at R lowers every probe inside R by ~ the same lapse factor and
leaves probes outside R at the anchor value. Their cone = deficit patterns that grow inward.
A deformation is separable only if its deficit pattern is NOT monotone in probe radius.
"""
import json, warnings
import numpy as np
from scipy.optimize import nnls
from core import kernel, residual_projector, whiten
from thin_shell import orbit_mass, shadow_mass, isco_mass
warnings.filterwarnings('ignore')
dm = 1e-5
Rs = np.unique(np.concatenate([np.linspace(3.2, 12, 60), np.logspace(np.log10(12), np.log10(2800), 60)]))
probe_resp = {
    'flare9': lambda R: (orbit_mass(9.0, dm, R) - 1) / dm,
    'isco_freq': lambda R: (isco_mass(dm, R) - 1) / dm,
    'shadow': lambda R: (shadow_mass(dm, R) - 1) / dm,
}
probe_kernel = {'flare9': lambda n: kernel(n, 'orbit', 9.0), 'isco_freq': lambda n: kernel(n, 'isco_freq'),
                'shadow': lambda n: kernel(n, 'shadow')}
cache = {p: np.array([f(R) for R in Rs]) for p, f in probe_resp.items()}
out = {}
for design in [('flare9', 'shadow'), ('isco_freq', 'shadow'), ('flare9', 'isco_freq', 'shadow')]:
    for n in [2, 3]:
        K = len(design) + 1
        u = np.array([kernel(n, 'orbit', 2800.0)] + [probe_kernel[p](n) for p in design])
        S = np.vstack([np.ones(len(Rs))] + [cache[p] for p in design])       # anchor response 1
        C = np.diag([0.005] + [0.05] * len(design)) ** 2
        Rp, Ci = residual_projector(np.ones((K, 1)), C); Wh = whiten(Ci)
        U = Wh @ Rp @ u; Sm = Wh @ Rp @ S
        best = 0
        for sign in (1, -1):
            c, res = nnls(Sm, sign * U)
            best = max(best, np.sqrt(max(0, 1 - res ** 2 / (U @ U))))
        key = f"{'+'.join(design)} n={n}"
        out[key] = best
        print(f"{key:32s}: max cos over all shell environments (R >= 3.2M) = {best:.4f}"
              f"   deficits rel. anchor: " + ", ".join(f"{p}={-(probe_kernel[p](n)-u[0]):.3f}" for p in design))
json.dump(out, open('results/env_monotone.json', 'w'), indent=1)
