"""Figures for the framework notes (static PNG, light surface, reference palette slots 1-3)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import kernel, confusion

S1, S2c, S3 = '#2a78d6', '#eb6834', '#1baf7a'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'lines.linewidth': 2})

# ---------------------------------------------------------------- Fig 1: radial sensitivity
fig, ax = plt.subplots(figsize=(7.2, 4.6))
from matplotlib.lines import Line2D
r_st = np.logspace(np.log10(6), np.log10(5000), 400); r_un = np.linspace(3, 6, 50)
for n, col in [(2, S1), (3, S2c)]:
    ax.plot(r_st, np.abs(kernel(n, 'orbit', r_st)), color=col)
    ax.plot(r_un, np.abs(kernel(n, 'orbit', r_un)), color=col, ls=':', lw=1.5)
    ax.plot([3], [abs(kernel(n, 'shadow'))], 'o', ms=9, color=col, mec=SURF, mew=2)
    ax.plot([6], [abs(kernel(n, 'isco_freq'))], 's', ms=9, color=col, mec=SURF, mew=2)
    ax.text(5200, abs(kernel(n, 'orbit', 5000.0)), f'  n = {n}', color=INK, fontsize=10, va='center')
handles = [Line2D([], [], color=INK2, label='orbit with r and Ω measured (dotted: unstable, r < 6M)'),
           Line2D([], [], marker='o', ls='', color=INK2, ms=8, label='shadow = eikonal ringdown ω_R (r = 3M)'),
           Line2D([], [], marker='s', ls='', color=INK2, ms=8, label='ISCO orbital frequency (r = 6M)')]
for rr, lab in [(9, 'GRAVITY flares\n≈ 9 M'), (2800, 'S2 pericentre\n≈ 2800 M')]:
    ax.axvline(rr, color=INK2, lw=1, ls=':')
    ax.text(rr * 1.08, 0.2, lab, fontsize=9, color=INK2, va='top')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(2.5, 12000); ax.set_ylim(1e-8, 1)
ax.set_xlabel('probe radius r / M'); ax.set_ylabel('|mass-bias kernel β|  (per unit δₙ)')
ax.set_title('Where each probe is sensitive to A(r) = 1 − 2M/r + δₙ(M/r)ⁿ', loc='left', fontsize=12)
ax.legend(handles=handles, frameon=False, loc='lower left', fontsize=9)
fig.tight_layout(); fig.savefig('figures/fig1_radial_kernels.png', dpi=180); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: environment vs gravity
fig, ax = plt.subplots(figsize=(7.2, 4.4))
sig_base = np.array([0.0043, 0.0064, 0.40, 0.20])
Am = np.ones((4, 1))
sf = np.logspace(-2.5, 0, 200)
for n, col in [(2, S1), (3, S2c)]:
    u = np.array([kernel(n, 'orbit', 2800.0)] * 2 + [kernel(n, 'orbit', 9.0), kernel(n, 'shadow')])
    env = np.array([1.0, 1.0, 9.0 / 2800.0, 0.0])
    cs = []
    for s in sf:
        sg = sig_base.copy(); sg[2] = s
        cs.append(confusion(Am, np.diag(sg ** 2), u, env))
    ax.plot(sf, cs, color=col, label=f'δ{"₂" if n == 2 else "₃"} vs extended mass')
    k = abs(kernel(n, 'orbit', 9.0) / kernel(n, 'shadow'))
    bound = 2 * np.sqrt(k) / (1 + k)
    ax.axhline(bound, color=col, lw=1, ls='--')
    ax.text(0.0035, bound + 0.004, f'bound 2√κ/(1+κ) = {bound:.3f}  (κ = {k:.2f})', fontsize=9, color=INK2)
ax.axvline(0.40, color=INK2, lw=1, ls=':'); ax.text(0.42, 0.835, 'flares\ntoday', fontsize=9, color=INK2)
ax.set_xscale('log'); ax.set_ylim(0.82, 1.005)
ax.set_xlabel('fractional precision of the 9 M flare-orbit mass'); ax.set_ylabel('|cos φ|  (1 = indistinguishable)')
ax.set_title('Equal-response lemma (environment outside both probes only)', loc='left', fontsize=12)
ax.legend(frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig('figures/fig2_environment_degeneracy.png', dpi=180); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: ringdown leakage (spectral)
res = json.load(open('results/qnm_validate.json'))
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
for ax, n in zip(axes, [2, 3]):
    for s_, col, lab in [(0, S1, 'scalar (s = 0)'), (1, S2c, 'electromagnetic (s = 1)'), (2, S3, 's = 2 proxy')]:
        rows = [x for x in res if x['n'] == n and x['s'] == s_]
        ax.plot([x['l'] for x in rows], [abs(x['leak_pct']) for x in rows], '-o', color=col, ms=8, mec=SURF, mew=2, label=lab)
    ax.set_xticks([2, 3, 4]); ax.set_xlabel('multipole ℓ'); ax.set_title(f'n = {n}', loc='left', fontsize=11)
axes[0].set_ylabel('|β_ωR − β_shadow| / |β_shadow|  (%)')
axes[1].legend(frameon=False, fontsize=9)
fig.suptitle('Ringdown frequency vs shadow: direct spectral QNMs', x=0.02, ha='left', fontsize=12)
fig.tight_layout(); fig.savefig('figures/fig3_ringdown_leakage.png', dpi=180); plt.close(fig)
print("figures written")
