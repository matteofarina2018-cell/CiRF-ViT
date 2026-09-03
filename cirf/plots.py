"""Plot primitives shared by the figure scripts.

Two visual styles, deliberately separated:

  `paper_style()`   the manuscript's look -- Helvetica (TeX Gyre Heros), the
                    per-model palette, axes of an exact physical size.
  `default_style()` plain matplotlib defaults, for the supplementary figures
                    where legibility matters more than house style.

Everything here is style and layout only; nothing computes a result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

from . import config as cfg

CM = 1.0 / 2.54

#: Metrically identical to Helvetica; falls back silently when not installed.
HELVETICA = 'TeX Gyre Heros'


def _register_fonts():
    """Pick up a locally installed TeX Gyre Heros, if there is one."""
    for pattern in ('texgyreheros-*.otf', 'texgyreheros-*.ttf'):
        for path in (Path.home() / '.fonts').glob(pattern):
            try:
                fm.fontManager.addfont(str(path))
            except Exception:
                pass
    return HELVETICA in {f.name for f in fm.fontManager.ttflist}


HAVE_HELVETICA = _register_fonts()
#: the manuscript's face when available, otherwise whatever sans-serif is
PAPER_FONT = HELVETICA if HAVE_HELVETICA else 'sans-serif'
_warned = False


def paper_style(font_size=15):
    """matplotlib defaults + the manuscript's typography.

    Falls back to the default sans-serif when TeX Gyre Heros is not installed;
    only the lettering changes, never the geometry of a panel.
    """
    global _warned
    if not HAVE_HELVETICA and not _warned:
        print(f'  (note: {HELVETICA} not installed -- figures use the default '
              f'sans-serif; install it for the manuscript lettering)')
        _warned = True
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({
        'font.family': PAPER_FONT,
        'font.size': font_size,
        'axes.linewidth': 1.1,
        'savefig.bbox': 'tight',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def default_style():
    """Plain matplotlib defaults (supplementary figures)."""
    matplotlib.rcdefaults()
    plt.rcParams.update(matplotlib.rcParamsDefault)


def save_figure(fig, path, dpi=300, close=True):
    """Write one PNG and report it. Figures in this repository are PNG only."""
    path = Path(path)
    if path.suffix.lower() not in ('.png', '.pdf'):
        path = path.with_suffix('.png')
    cfg.ensure(path.parent)
    fig.savefig(path, dpi=dpi)
    if close:
        plt.close(fig)
    print(f'  saved {path}')
    return path


def fixed_axes(width_cm, height_cm=5.0, ml=0.55, mr=0.07, mb=0.42, mt=0.07):
    """A figure whose *axes area* has an exact physical size, in cm.

    The manuscript places several panels side by side at the same axis height,
    so the axes -- not the figure -- must match. Margins are in inches.
    """
    wa, ha = width_cm * CM, height_cm * CM
    fig_w, fig_h = ml + wa + mr, mb + ha + mt
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([ml / fig_w, mb / fig_h, wa / fig_w, ha / fig_h])
    return fig, ax


def despine(ax, which=('top', 'right')):
    for side in which:
        if side in ax.spines:
            ax.spines[side].set_visible(False)


def style_axes(ax, tick_fs=None, spine_lw=1.0, tick_len=4.0):
    despine(ax)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(spine_lw)
    ax.tick_params(axis='both', length=tick_len, width=spine_lw,
                   **({'labelsize': tick_fs} if tick_fs else {}))


# -----------------------------------------------------------------------------
# Statistics shown on figures
# -----------------------------------------------------------------------------

def p_to_stars(p, ns='ns'):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ns


def wilcoxon_vs_reference(box, order, ref='transformer'):
    """Two-sided Wilcoxon signed-rank of each model against `ref`.

    The comparison is paired neuron by neuron, which is what makes it a test of
    the architecture rather than of the population: every model predicts the
    same cells. No multiple-comparison correction is applied -- the tests are a
    fixed, pre-specified set (each reference model against the proposed one).
    """
    from scipy.stats import wilcoxon

    i_ref = order.index(ref)
    ref_vals = np.asarray(box[i_ref], dtype=float)
    out = {}
    for i, m in enumerate(order):
        if i == i_ref:
            continue
        try:
            out[m] = float(wilcoxon(ref_vals, np.asarray(box[i], dtype=float)).pvalue)
        except ValueError:       # all differences zero -> no evidence either way
            out[m] = 1.0
    return out


def add_significance_stars(ax, box, order, ref='transformer', fontsize=8,
                           dy=0.025, verbose=True):
    """Wilcoxon stars above every model except the reference."""
    pvals = wilcoxon_vs_reference(box, order, ref)
    top = max(float(np.percentile(np.asarray(b, dtype=float), 95)) for b in box)
    for i, m in enumerate(order):
        if m == ref:
            continue
        stars = p_to_stars(pvals[m])
        ax.text(i, top + dy, stars, ha='center', va='bottom',
                fontsize=fontsize, color='black', zorder=5)
        if verbose:
            print(f'  [wilcoxon] {ref} vs {m}: p={pvals[m]:.3e}  ->  {stars}')
    return pvals


# -----------------------------------------------------------------------------
# Recurring panels
# -----------------------------------------------------------------------------

def boxplot(ax, box, colors, positions=None, width=0.62, points=True,
            centre='median', lw=1.0, point_size=30, seed=0):
    """Box (IQR) with 5-95 whiskers, a thick line at the centre, and points.

    `centre` picks what that line marks -- 'median' or 'mean' -- and the box's
    own median line is hidden so the figure never shows two unlabelled lines.
    """
    if positions is None:
        positions = np.arange(len(box))
    bp = ax.boxplot(box, positions=positions, widths=width, patch_artist=True,
                    showfliers=False, whis=(5, 95),
                    medianprops=dict(linewidth=0),
                    whiskerprops=dict(color='0.55', linewidth=1.1 * lw),
                    capprops=dict(color='0.55', linewidth=1.1 * lw),
                    boxprops=dict(linewidth=1.6 * lw))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.28)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.8 * lw)
    stat = np.median if centre == 'median' else np.mean
    for i, (v, c) in enumerate(zip(box, colors)):
        mu = float(stat(np.asarray(v, dtype=float)))
        ax.plot([positions[i] - width * 0.58, positions[i] + width * 0.58], [mu, mu],
                color=c, linewidth=3.4 * lw, solid_capstyle='round', zorder=4)
    if points:
        rng = np.random.default_rng(seed)
        for i, (v, c) in enumerate(zip(box, colors)):
            v = np.asarray(v, dtype=float)
            jitter = rng.uniform(-0.13, 0.13, size=v.size)
            ax.scatter(positions[i] + jitter, v, s=point_size, color=c,
                       edgecolor='black', linewidth=0.4, alpha=0.85, zorder=3)
    return bp


def model_colors(order, palette=None):
    palette = palette or cfg.PAPER_COLORS
    return [palette[m] for m in order]
