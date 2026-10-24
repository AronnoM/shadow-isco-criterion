"""Paper versions of the figures: column width (~3.6 in), vector PDF, no headline titles (captions carry them).
Every text, line and marker is scaled with the figure so the design of make_figures*.py is preserved."""
import os, runpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.axes, matplotlib.figure, matplotlib.text, matplotlib.lines, matplotlib.pyplot as plt

OUT = os.environ.get('PAPER_FIG_DIR', '../paper/figures')
os.makedirs(OUT, exist_ok=True)
SCALE = 0.52            # 7.2 in -> 3.7 in
TEXT = 0.66             # 11 pt -> 7.3 pt, 9 pt -> 6 pt

_title = matplotlib.axes.Axes.set_title
def set_title(self, *a, **k):          # drop headline titles (fontsize 12), keep panel labels
    if k.get('fontsize', 12) < 12:
        return _title(self, *a, **k)
matplotlib.axes.Axes.set_title = set_title
matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None

_subplots = plt.subplots
def subplots(*a, **k):
    if 'figsize' in k:
        w, h = k['figsize']; k['figsize'] = (w * SCALE, h * SCALE * 1.15)
    return _subplots(*a, **k)
plt.subplots = subplots

_save = matplotlib.figure.Figure.savefig
def save(self, fname, *a, **k):
    for t in self.findobj(matplotlib.text.Text):
        t.set_fontsize(max(5.0, t.get_fontsize() * TEXT))
    for ln in self.findobj(matplotlib.lines.Line2D):
        ln.set_linewidth(ln.get_linewidth() * 0.65)
        ln.set_markersize(ln.get_markersize() * 0.6)
        ln.set_markeredgewidth(ln.get_markeredgewidth() * 0.6)
    base = os.path.splitext(os.path.basename(str(fname)))[0]
    for ax in self.axes:
        ax.tick_params(width=0.5, length=2.5, labelsize=7)
        if base.startswith('fig4'):
            leg = ax.get_legend()
            if leg is not None:
                leg._loc = 7                     # centre right, clear of the curves
            for t in ax.texts:
                if t.get_text().startswith('ISCO'):
                    t.set_x(5.7); t.set_horizontalalignment('right')
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    self.tight_layout(pad=0.3)
    k.pop('dpi', None)
    _save(self, os.path.join(OUT, base + '.pdf'), *a, **k)
matplotlib.figure.Figure.savefig = save

for s in ['make_figures.py', 'make_figures_env.py']:
    runpy.run_path(s, run_name='__main__')
print('paper figures written to', OUT)
