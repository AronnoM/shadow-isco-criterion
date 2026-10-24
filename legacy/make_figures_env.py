"""Stage-2 figures: thin-shell response kernels and the S2 constraint on shells."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S1, S2c, S3 = '#2a78d6', '#eb6834', '#1baf7a'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'lines.linewidth': 2})
d = json.load(open('results/env_theorem.json'))
resp = np.array(d['response_grid'])
R, a, f, s = resp.T

# Fig 4: response relative to the S2 anchor
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for lo, hi in [(3, 9), (9, 3000)]:
    sel = (R > lo) & (R < hi)
    ax.plot(R[sel], (f - a)[sel], color=S1, label='flare orbit at 9 M (ray-traced)' if lo == 3 else None)
    ax.plot(R[sel], (s - a)[sel], color=S2c, label='shadow' if lo == 3 else None)
Rl = np.logspace(np.log10(9.01), np.log10(3000), 200)
ax.plot(Rl, -(Rl - 3) / (Rl - 2), color=INK2, lw=1, ls='--', label='lapse only: −(R−3M)/(R−2M)')
ax.axvline(9, color=INK2, lw=1, ls=':'); ax.axvline(6, color=INK2, lw=1, ls=':')
ax.text(9.4, -0.1, 'flare\nradius', fontsize=9, color=INK2); ax.text(6.2, -0.1, 'ISCO', fontsize=9, color=INK2)
ax.set_xscale('log'); ax.set_ylim(-1.08, 0.05)
ax.set_xlabel('shell radius R / M'); ax.set_ylabel('inferred-mass shift relative to S2, per unit m/M')
ax.set_title('A thin shell: equal response only when it lies outside both probes', loc='left', fontsize=12)
ax.legend(frameon=False, loc='lower right', fontsize=9)
fig.tight_layout(); fig.savefig('figures/fig4_thin_shell_response.png', dpi=180); plt.close(fig)

# Fig 6: S2 constraint on shells (GRAVITY precession equivalence + full Keck orbit refit)
lim = np.array(d['s2_shell_limits']); s2 = d['s2']
rf = json.load(open('results/s2_refit.json'))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
sel = (lim[:, 0] > s2['r_peri'] * 1.001) & (lim[:, 0] < s2['r_apo'])
ax.plot(lim[sel, 0], 100 * lim[sel, 2], color=S1, label='GRAVITY precession equivalence, 2σ (indicative)')
sh = [x for x in rf['shells'] if np.isfinite(x['sigma_m'])]
ax.plot([x['R_M'] for x in sh], [100 * 1.645 * x['sigma_m'] for x in sh], color=S3,
        label='Keck orbit refit, 95% sensitivity (14 parameters profiled)')
rp_M = rf['rp_AU'] * rf['AU_in_M']
ax.axvspan(1, rp_M, color=S3, alpha=0.10)
ax.text(12, 0.025, 'inside S2 pericentre:\nexactly degenerate with M\n(refit: σ = ∞)', fontsize=9, color=INK2)
mim = d['mimic_d2_0.5']
ax.plot([mim['R_far'], mim['R_in']], [100 * mim['m_far'], 100 * mim['m_in']], 'o', ms=9, color=S2c, mec=SURF, mew=2,
        label='two shells that mimic δ₂ = 0.5 (numerical)')
ax.axvline(rp_M, color=INK2, lw=1, ls=':'); ax.axvline(rf['ra_AU'] * rf['AU_in_M'], color=INK2, lw=1, ls=':')
ax.text(rp_M * 1.1, 0.013, 'S2 peri', fontsize=9, color=INK2)
ax.text(rf['ra_AU'] * rf['AU_in_M'] * 1.1, 0.013, 'S2 apo', fontsize=9, color=INK2)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(3, 2e5); ax.set_ylim(0.01, 100)
ax.set_xlabel('shell radius R / M'); ax.set_ylabel('shell mass (% of M)')
ax.set_title('Stellar orbits cannot see the environment that mimics a deformation', loc='left', fontsize=12)
ax.legend(frameon=False, loc='upper right', fontsize=9)
fig.tight_layout(); fig.savefig('figures/fig6_s2_shell_limits.png', dpi=180); plt.close(fig)
print('ok')
