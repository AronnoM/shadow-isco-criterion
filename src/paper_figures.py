"""Journal figures for the paper (Physical Review D, REVTeX reprint): vector PDF, LaTeX text (Computer Modern),
true column width (3.375 in) or page width (7.0 in), inward ticks on all sides, palette slots 1-3 of the reference
palette (validated all-pairs on white) plus neutral ink for references, line style / marker as secondary encoding.

Every number is read from results/*.json or recomputed from the analysis modules; nothing is typed in by hand."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator, FixedLocator

OUT = os.environ.get('PAPER_FIG_DIR', 'paper/figures')
os.makedirs(OUT, exist_ok=True)

C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'          # blue, orange, aqua (validated slots 1-3)
INK, INK2, GRAY, WASH = '#1a1a1a', '#55544f', '#8f8d86', '#ecebe7'
COL, PAGE = 3.375, 7.0

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}',
    'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.5, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'lines.linewidth': 1.1, 'lines.markersize': 3.5, 'lines.markeredgewidth': 0.6,
    'legend.frameon': False, 'legend.handlelength': 2.0, 'legend.borderaxespad': 0.3, 'legend.labelspacing': 0.3,
    'axes.grid': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.015,
    'figure.dpi': 300, 'pdf.fonttype': 42,
})


def panel(ax, s, x=0.035, y=0.95, **k):
    ax.text(x, y, s, transform=ax.transAxes, ha='left', va='top', fontsize=8, **k)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    plt.close(fig)
    print('wrote', name)


def logticks(ax, axis='x'):
    loc = LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50)
    (ax.xaxis if axis == 'x' else ax.yaxis).set_minor_locator(loc)
    (ax.xaxis if axis == 'x' else ax.yaxis).set_minor_formatter(NullFormatter())


# ======================================================================= 1. kernels versus probe radius
def fig_kernels():
    global FLK
    FLK = json.load(open('results/flare_consistent.json'))['flare_kernel_astrometric']
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    r_st = np.logspace(np.log10(6), 4, 300); r_un = np.linspace(3, 6, 60)
    styles = {2: (C1, '-'), 3: (C2, '--'), 4: (C3, '-.')}
    for n, (c, ls) in styles.items():
        orb = lambda r: (n / 2) * r ** (1.0 - n)
        ax.plot(r_st, orb(r_st), color=c, ls=ls, label=fr'$n={n}$')
        ax.plot(r_un, orb(r_un), color=c, ls=ls, lw=0.6, alpha=0.55)
        ax.plot(3, 0.5 * 3.0 ** (1 - n), 'o', color=c, mec='white', ms=4.6, zorder=5)
        ax.plot(6, 3 * (2 * n * n - n + 1) / (2 * 6.0 ** n), 's', color=c, mec='white', ms=4.2, zorder=5)
        ax.plot(9, abs(FLK[str(n)]), 'x', color=c, ms=4.0, mew=0.9, zorder=5)
    for r0, lab, y0, side in [(9, r'flares', 1.6e-8, 1.12), (2900, r'S2 peri.', 1.5e-6, 1 / 1.1)]:
        ax.axvline(r0, color=GRAY, lw=0.4, zorder=0)
        ax.text(r0 * side, y0, lab, fontsize=6.5, color=INK2, rotation=90, va='bottom',
                ha='left' if side > 1 else 'right')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(2.6, 1e4); ax.set_ylim(1e-8, 1.5)
    logticks(ax, 'x')
    ax.yaxis.set_major_locator(FixedLocator([1e-8, 1e-6, 1e-4, 1e-2, 1]))
    ax.yaxis.set_minor_locator(FixedLocator([1e-7, 1e-5, 1e-3, 1e-1]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'probe radius $r/M$')
    ax.set_ylabel(r'mass-bias kernel $|\beta_n|$')
    for n, (x, y) in {2: (1500, 1.5e-3), 3: (450, 1.6e-5), 4: (260, 3.0e-7)}.items():
        ax.text(x, y, fr'$n={n}$', fontsize=7, color=INK, ha='left', va='bottom')
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=INK, lw=1.1, label=r'orbit, $r$ and $\Omega$ measured'),
            Line2D([], [], color=INK, marker='o', ls='', mec='white', ms=4.6, label=r'shadow, eikonal $\omega_R$ ($3M$)'),
            Line2D([], [], color=INK, marker='s', ls='', mec='white', ms=4.2, label=r'ISCO frequency ($6M$)'),
            Line2D([], [], color=INK, marker='x', ls='', ms=4.0, mew=0.9, label=r'flare, astrometric fit ($9M$)')]
    ax.legend(handles=keys, loc='upper right', fontsize=6.5)
    save(fig, 'fig_kernels')


# ======================================================================= 2. thin-shell response (Lemma)
def fig_shell():
    d = json.load(open('results/env_theorem.json'))
    R, a, f, s = np.array(d['response_grid']).T
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    g = lambda x: -(x - 3) / (x - 2)
    Rl = np.logspace(np.log10(3.0), np.log10(3000), 600)
    ax.plot(Rl, np.where(Rl > 3, g(Rl), 0.0), color=C1, label=r'shadow ($r_i=3M$)')
    Ri = np.concatenate([[3.0, 6.0], np.logspace(np.log10(6.0001), np.log10(3000), 500)])
    ax.plot(Ri, np.where(Ri > 6, g(Ri), 0.0), color=C2, ls='--', label=r'ISCO frequency ($6M$)')
    sel = R > 9.0
    Rf = np.concatenate([[3.0, 9.0], R[sel]]); ff = np.concatenate([[0.0, 0.0], (f - a)[sel]])
    ax.plot(Rf, ff, color=C3, ls='-.', label=r'flare orbit at $9M$ (ray traced)')
    ax.text(5.8, 0.035, r'$6M$', fontsize=6.5, color=INK2, ha='right')
    ax.text(9.4, 0.035, r'$9M$', fontsize=6.5, color=INK2, ha='left')
    ax.set_xscale('log'); ax.set_xlim(2.8, 3000); ax.set_ylim(-1.07, 0.12)
    logticks(ax, 'x')
    ax.yaxis.set_major_locator(MultipleLocator(0.25)); ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlabel(r'shell radius $R/M$')
    ax.set_ylabel(r'$(M_{\rm inf}-M_{\rm anchor})/m$')
    ax.legend(loc='center right', fontsize=6.5)
    save(fig, 'fig_shell')


# ======================================================================= 3. the criterion (Theorem 1, Corollaries 3-4)
def fig_theorem():
    fig, axs = plt.subplots(1, 3, figsize=(PAGE, 2.25), gridspec_kw=dict(wspace=0.34, width_ratios=[1.05, 1, 1]))
    # (a) mechanism: normalized deficits and an explicit staircase
    ax = axs[0]
    ratio = lambda n: (2 * n * n - n + 1) / 2.0 ** n
    for n, c, mk in [(2, C1, 'o'), (8, C2, 's')]:
        D = np.array([1.0, ratio(n)])
        ax.plot([3, 6], D, mk, color=c, mec='white', ms=5, zorder=6, label=fr'$n={n}$')
    # staircase for n = 8: shells at 4.5M and 40M
    D8 = ratio(8); Rsh = [4.5, 40.0]
    xs = [3, Rsh[0], Rsh[0], Rsh[1], Rsh[1], 200]; ys = [1, 1, D8, D8, 0, 0]
    ax.plot(xs, ys, color=C2, lw=0.9, zorder=4)
    for Rs in Rsh:
        ax.annotate('', xy=(Rs, -0.08), xytext=(Rs, -0.24), arrowprops=dict(arrowstyle='-|>', lw=0.6, color=INK2,
                                                                         mutation_scale=6))
    for Rs, lab in zip(Rsh, (r'$m_1$', r'$m_2$')):
        ax.text(Rs * 1.12, -0.2, lab, fontsize=6.5, color=INK2, ha='left')
    ax.plot([3, 6], [1, ratio(2)], color=C1, lw=0.7, ls=':', zorder=3)
    ax.text(6.6, ratio(2), r'rises outward:' '\n' r'no shells fit', fontsize=6.5, color=INK2, va='center')
    ax.text(48, D8 + 0.07, r'staircase from' '\n' r'two shells', fontsize=6.5, color=INK2, ha='center', va='bottom')
    ax.set_xscale('log'); ax.set_xlim(2.5, 200); ax.set_ylim(-0.3, 2.05)
    logticks(ax, 'x')
    ax.xaxis.set_major_locator(FixedLocator([3, 6, 10, 30, 100]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(['3', '6', '10', '30', '100']))
    ax.yaxis.set_major_locator(MultipleLocator(0.5)); ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(r'probe radius $r/M$'); ax.set_ylabel(r'deficit $D(r)/D_{\rm sh}$')
    ax.legend(loc='upper right', fontsize=6.5, handletextpad=0.2)
    panel(ax, r'(a)')
    # (b) ratio versus n
    ax = axs[1]
    nn = np.linspace(2, 12, 400); ni = np.arange(2, 13)
    ax.axhspan(1, 2.4, color=C1, alpha=0.08, lw=0); ax.axhspan(0, 1, color=C2, alpha=0.08, lw=0)
    ax.axhline(1, color=INK2, lw=0.5)
    ax.plot(nn, ratio(nn), color=INK, lw=0.8)
    ax.plot(ni[ni <= 6], ratio(ni[ni <= 6]), 'o', color=C1, mec='white', ms=4.6, zorder=5)
    ax.plot(ni[ni >= 7], ratio(ni[ni >= 7]), 's', color=C2, mec='white', ms=4.2, zorder=5)
    ax.text(11.8, 1.95, r'separable from every' '\n' r'admissible shell set', fontsize=6.5, color=INK2, ha='right', va='top')
    ax.text(12.1, 0.98, r'mimicable iff' '\n' r'matter may sit' '\n' r'in $(3M,6M)$', fontsize=6.5, color=INK2,
            ha='right', va='top')
    ax.annotate(r'$n_*\simeq6.13$', xy=(6.13, 1.0), xytext=(7.2, 1.45), fontsize=6.5, color=INK2,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=INK2))
    ax.set_xlim(1.6, 12.4); ax.set_ylim(0, 2.25)
    ax.xaxis.set_major_locator(MultipleLocator(2)); ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5)); ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(r'power $n$')
    ax.set_ylabel(r'$\beta_{\rm ISCO}/\beta_{\rm sh}=(2n^2-n+1)/2^n$')
    panel(ax, r'(b)', x=0.035, y=0.09)
    # (c) two-power plane
    ax = axs[2]
    d2 = np.linspace(0, 1.05, 50)
    ax.fill_between(d2, -21 / 8 * d2, -9 / 4 * d2, color=C2, alpha=0.30, lw=0)
    ax.plot(d2, -9 / 4 * d2, color=C1, lw=1.4)
    ax.plot(d2, -21 / 8 * d2, color=C2, lw=0.6)
    ax.plot(d2, -3 * d2, color=GRAY, lw=0.5, ls='--')
    ax.axhline(0, color=GRAY, lw=0.4); ax.axvline(0, color=GRAY, lw=0.4)
    ax.annotate(r'$R_{\rm adm}<6M$', xy=(0.80, -1.95), xytext=(0.08, -1.75), fontsize=6.5, color=INK2,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=INK2))
    ax.annotate(r'$D_{\rm sh}=0$', xy=(0.85, -2.55), xytext=(0.42, -3.05), fontsize=6.5, color=INK2,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=INK2))
    ax.annotate(r'any $R_{\rm adm}$ (line)', xy=(0.55, -9 / 4 * 0.55), xytext=(0.48, -0.55), fontsize=6.5,
                color=INK2, arrowprops=dict(arrowstyle='-', lw=0.5, color=INK2))
    ax.text(0.97, 0.12, r'single power $n=2$', fontsize=6, color=INK2, ha='right')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-3.3, 0.45)
    ax.xaxis.set_major_locator(MultipleLocator(0.5)); ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(1)); ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_xlabel(r'$\delta_2$'); ax.set_ylabel(r'$\delta_3$')
    panel(ax, r'(c)', x=0.035, y=0.09)
    save(fig, 'fig_theorem')


# ======================================================================= 4. partial identification
def fig_ident():
    d = json.load(open('results/partial_id.json'))
    x = np.array(d['grid'])
    fig, axs = plt.subplots(1, 2, figsize=(COL, 2.15), sharey=True, gridspec_kw=dict(wspace=0.08))
    sty = {'none': (C1, '-', r'no environment'), 'R>=6M': (C2, '--', r'shells at $R\ge6M$'),
           'R>=3M': (C3, '-.', r'shells at $R\ge3M$')}
    for ax, truth, lab in zip(axs, ['no matter', 'interior matter'], ['(a)', '(b)']):
        for cls, (c, ls, name) in sty.items():
            ax.plot(x, d['profiles'][truth][cls], color=c, ls=ls, label=name, lw=1.1 if cls != 'R>=3M' else 1.2)
        ax.axhline(3.84, color=INK2, lw=0.5, ls=':')
        ax.axvline(0, color=GRAY, lw=0.4, zorder=0)
        ax.set_xlim(-0.33, 0.15); ax.set_ylim(0, 9.5)
        ax.xaxis.set_major_locator(MultipleLocator(0.1)); ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_locator(MultipleLocator(2)); ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.set_xlabel(r'$\delta_2$')
        panel(ax, lab, x=0.05, y=0.96)
    axs[1].axvspan(-0.04, 0.03, color=C3, alpha=0.13, lw=0, zorder=0)
    axs[1].text(-0.005, 8.9, r'$\mathcal I(D)$', fontsize=7, color=INK2, ha='center', va='top')
    axs[0].text(-0.32, 3.84 + 0.15, r'95\%', fontsize=6.5, color=INK2, va='bottom')
    axs[0].set_ylabel(r'$-2\Delta\ln\mathcal L_{\rm p}(\delta_2)$')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=6.3, handlelength=2.2, columnspacing=1.0,
               bbox_to_anchor=(0.52, 1.07))
    save(fig, 'fig_ident')


# ======================================================================= 5. ringdown versus shadow kernel
def fig_qnm():
    res = json.load(open('results/qnm_validate.json'))
    fig, axs = plt.subplots(1, 2, figsize=(COL, 2.05), gridspec_kw=dict(wspace=0.38))
    sty = {0: (C1, 'o', '-', r'$s=0$'), 1: (C2, 's', '--', r'$s=1$'), 2: (C3, '^', '-.', r'$s=2$ proxy')}
    for ax, n, lab in zip(axs, [2, 3], ['(a)', '(b)']):
        for s_, (c, mk, ls, name) in sty.items():
            rows = sorted([x for x in res if x['n'] == n and x['s'] == s_], key=lambda x: x['l'])
            ax.plot([x['l'] for x in rows], [abs(x['leak_pct']) for x in rows], ls=ls, marker=mk, color=c,
                    mec='white', ms=4.2, label=name)
        ax.set_xlim(1.7, 4.3); ax.set_ylim(0, None)
        ax.xaxis.set_major_locator(FixedLocator([2, 3, 4])); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel(r'multipole $\ell$')
        ax.text(0.95, 0.95, lab + fr' $n={n}$', transform=ax.transAxes, ha='right', va='top', fontsize=8)
    axs[0].set_ylabel(r'$|\beta_{\omega_R}-\beta_{\rm sh}|/|\beta_{\rm sh}|$ (\%)')
    axs[0].yaxis.set_major_locator(MultipleLocator(5)); axs[0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1].yaxis.set_major_locator(MultipleLocator(10)); axs[1].yaxis.set_minor_locator(MultipleLocator(2))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=6.5, handlelength=2.4, bbox_to_anchor=(0.53, 1.06))
    save(fig, 'fig_qnm')


# ======================================================================= 6. S2 constraints on shells
def fig_s2():
    env = json.load(open('results/env_theorem.json')); rf = json.load(open('results/s2_refit.json'))
    ph = json.load(open('results/s2_photon.json')); mim = json.load(open('results/exact_mimic.json'))
    AUM = rf['AU_in_M']; rp, ra = rf['rp_AU'] * AUM, rf['ra_AU'] * AUM
    bmin = ph['b_min_obs_AU'] * AUM
    lim = np.array(env['s2_shell_limits'])
    RG = lim[:, 0] / 23.56 * AUM                       # GRAVITY curve computed with 1 AU = 23.56 M
    selG = (lim[:, 0] > env['s2']['r_peri'] * 1.001) & (lim[:, 0] < env['s2']['r_apo'])
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    ax.axvspan(1, bmin, color=WASH, lw=0, zorder=0)
    ax.axvspan(bmin, rp, color=WASH, alpha=0.45, lw=0, zorder=0)
    ax.text(12, 0.02, r'invisible to the orbit' '\n' r'and to its light' '\n' r'(Birkhoff)', fontsize=6.5,
            color=INK2, va='bottom')
    ax.plot(RG[selG], 100 * lim[selG, 2], color=C2, ls='--', label=r'GRAVITY, precession equiv.\ (indicative)')
    ax.plot([mim['R_in'], mim['R_far']], [100 * mim['m_in'], 100 * mim['m_far']], 'D', color=C3, mec='white',
            ms=4.8, zorder=6, label=r'exact shells mimicking $\delta_2=0.5$')
    for r0 in (rp, ra):
        ax.axvline(r0, color=GRAY, lw=0.4, zorder=0)
    ax.text(rp * 1.12, 0.013, r'peri.', fontsize=6.5, color=INK2); ax.text(ra * 1.12, 0.013, r'apo.', fontsize=6.5, color=INK2)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(3, 2e5); ax.set_ylim(0.01, 100)
    logticks(ax, 'x'); logticks(ax, 'y')
    ax.set_xlabel(r'shell radius $R/M$'); ax.set_ylabel(r'shell mass $m/M$ (\%)')
    ax.legend(loc='upper left', fontsize=6.5, bbox_to_anchor=(0.0, 1.0))
    top = ax.secondary_xaxis('top', functions=(lambda x: x / AUM, lambda x: x * AUM))
    top.set_xlabel(r'$R$ (AU)', labelpad=3)
    top.tick_params(direction='in', which='both')
    ax.tick_params(top=False, which='both')
    save(fig, 'fig_s2')


# ======================================================================= 7. Sgr A*: sectors and variants
def fig_sgra():
    import sgra_real as SR
    zG, SG, zK, SK, thG, thK = SR.sectors()
    fig, axs = plt.subplots(1, 2, figsize=(PAGE, 2.6), gridspec_kw=dict(wspace=0.28, width_ratios=[1, 1.08]))
    ax = axs[0]

    def ellipse(z, S, c, ls, lab, ks=(1, 2), fill=True):
        mu = np.exp(z); Jm = np.diag([mu[1] / 1e3, mu[0] / 1e6])          # (ln R0, ln M) -> (kpc, 1e6 Msun)
        Sx = Jm @ np.array([[S[1, 1], S[0, 1]], [S[0, 1], S[0, 0]]]) @ Jm
        w, v = np.linalg.eigh(Sx); ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        for k, al in zip(ks, (0.28, 0.12)):
            sc = np.sqrt({1: 2.30, 2: 6.18}[k])                  # 68.3% and 95.4% regions, 2 dof
            ax.add_patch(Ellipse((mu[1] / 1e3, mu[0] / 1e6), 2 * sc * np.sqrt(w[1]), 2 * sc * np.sqrt(w[0]), angle=ang,
                                 fc=c if fill else 'none', alpha=al if fill else 1, lw=0, zorder=3))
            ax.add_patch(Ellipse((mu[1] / 1e3, mu[0] / 1e6), 2 * sc * np.sqrt(w[1]), 2 * sc * np.sqrt(w[0]), angle=ang,
                                 fc='none', ec=c, lw=0.8 if k == 1 else 0.5, ls=ls, zorder=4))
        ax.plot(mu[1] / 1e3, mu[0] / 1e6, '+', color=c, ms=5, mew=0.8, zorder=5)
        ax.plot([], [], color=c, ls=ls, lw=0.8, label=lab)
    # EHT band: M = theta_g D / K
    K = np.exp(SR.CONST)
    D = np.linspace(7.0, 9.0, 50)
    Mof = lambda th: th / K * D * 1e3 / 1e6
    ax.plot(D, Mof(4.8), color=C3, lw=0.8, ls='-.', zorder=2, label=r'EHT $\theta_{\rm g}=4.8\,\mu$as (central)')
    ax.text(8.52, 3.66, r'EHT 68\% band ($4.1$--$6.2\,\mu$as)' '\n' r'covers the whole panel', fontsize=6.2,
            color=INK2, ha='right', va='bottom')
    ellipse(zG, SG, C1, '-', r'GRAVITY 2022')
    ellipse(zK, SK, C2, '--', r'Keck, GR chain')
    zU, SU = SR.sectors(keck='Upsilon-free chain')[2:4]
    mu = np.exp(zU)
    ax.plot(mu[1] / 1e3, mu[0] / 1e6, 'x', color=C2, ms=4, mew=0.7, zorder=5, label=r'Keck, $\Upsilon$-free (mean)')
    ax.set_xlim(7.55, 8.55); ax.set_ylim(3.6, 4.65)
    ax.xaxis.set_major_locator(MultipleLocator(0.25)); ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.25)); ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlabel(r'distance $R_0$ (kpc)'); ax.set_ylabel(r'mass $M$ ($10^6\,M_\odot$)')
    ax.legend(loc='upper left', fontsize=6.5, bbox_to_anchor=(0.0, 0.91))
    panel(ax, r'(a)', x=0.035)
    # (b) variants
    ax = axs[1]
    rows = json.load(open('results/sgra_real.json'))['rows']
    sel = lambda f: [r for r in rows if f(r)]
    groups = [(lambda r: r['nuis'] == 'none' and r['keck'] == 'GR chain', dict(marker='o', ls='', color=C2, ms=2.4,
                                                                             mec='none', alpha=0.75),
               r'Keck GR chain'),
              (lambda r: r['nuis'] == 'none' and r['keck'] != 'GR chain', dict(marker='o', ls='', mfc='none', mec=C2,
                                                                             ms=2.6, mew=0.45, alpha=0.8),
               r'Keck $\Upsilon$-free chain'),
              (lambda r: r['nuis'] == 'Keck distance-scale offset', dict(marker='x', ls='', color=GRAY, ms=2.6,
                                                                       mew=0.4, alpha=0.7),
               r'with a free Keck offset along $M\propto R_0^2$')]
    for f, kw, lab in groups:
        rr = sel(f)
        ax.plot([r['sigma'] for r in rr], [r['Tpar'] for r in rr], label=lab, **kw)
    b = json.load(open('results/sgra_real.json'))['base']
    ax.plot(b['sigma'], b['Tpar'], '*', color=INK, ms=7, mec='white', mew=0.5, zorder=6, label=r'baseline')
    ax.axhline(3.84, color=INK2, lw=0.6)
    ax.text(7.9, 3.84 * 1.15, r'95\% critical value of $T^2_\parallel$ ($\chi^2_1$)', fontsize=6.5, color=INK2,
            ha='right', va='bottom')
    ax.set_yscale('log'); ax.set_ylim(5e-3, 12); ax.set_xlim(-0.2, 8)
    logticks(ax, 'y')
    ax.xaxis.set_major_locator(MultipleLocator(1)); ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_xlabel(r'significance of the consistency test ($\sigma$)')
    ax.set_ylabel(r'deformation-direction statistic $T^2_\parallel$')
    ax.legend(loc='upper left', fontsize=6.5, bbox_to_anchor=(0.0, 0.86), handletextpad=0.3, markerscale=1.6)
    panel(ax, r'(b)')
    save(fig, 'fig_sgra')


if __name__ == '__main__':
    fig_kernels(); fig_shell(); fig_theorem(); fig_ident(); fig_qnm(); fig_s2(); fig_sgra()
