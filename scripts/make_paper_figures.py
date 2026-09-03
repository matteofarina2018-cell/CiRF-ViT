#!/usr/bin/env python
"""Regenerate the paper's figures from trained checkpoints.

    python scripts/make_paper_figures.py                     # every figure, both datasets
    python scripts/make_paper_figures.py --figures metrics_ev --dataset mouse
    python scripts/make_paper_figures.py --data my.npz

Figures (each written per dataset to results/figures/paper/):

    metrics_ev        explained variance of LN / GP / CNN / CiRF-ViT, with the
                      paired Wilcoxon test of each reference against CiRF-ViT
    scatter_ev        the same, neuron by neuron, CiRF-ViT against the CNN
    fraction_ev       explained variance against the share of training data
    metrics_lsta_corr agreement with the experimental LSTA, same four models
    ablation          one architectural component removed at a time, plus the
                      two controls that separate the crop's effect on the
                      attention from its effect on the readout
    focal_comparison  the measured crop against ViT-P's learned focal bias
    crop_vs_nocrop    training and validation loss with and without the crop

Inputs, all produced by earlier scripts and never recomputed here:

    results/<ds>/checkpoints/<model>/eval/eval_data.npz   scripts/evaluate.py
    results/<ds>/fraction_study/...                       scripts/fraction_study.py
    results/<ds>/experiments/ablation.json                scripts/ablation.py
    results/<ds>/experiments/history_nocrop.json          scripts/crop_vs_nocrop.py

Every run of the fraction study is used: a curve is the mean over all of them
and its band is their spread, and no figure ever drops a run. A figure whose
inputs are missing is skipped with a message naming the script that produces
them.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from _common import (add_dataset_arg, add_device_arg, banner, cfg,
                     datasets_from, device_from)
from cirf import checkpoints as ckpt
from cirf import data as dat
from cirf import evaluate as ev
from cirf import lsta as L
from cirf import plots
from cirf.metrics import explained_variance_per_neuron

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = cfg.PAPER_COLORS
ABBR = cfg.MODEL_ABBR
ORDER = ('LN', 'GP', 'cnn', 'transformer')
CURVE_MODELS = ('GP', 'cnn', 'transformer')

# -- geometry of the small panels: the axes are 5 cm tall in the manuscript ----
H_CM = 5.0
W_FRACTION, W_SCATTER, W_METRICS = H_CM * 1.343, H_CM * 1.000, H_CM * 1.276
YLIM, YTICKS = (0.0, 1.12), np.arange(0.0, 1.01, 0.2)
MB, MT = 0.42, 0.07                       # shared vertical margins, inches
FIGH_CM = (MB + H_CM * plots.CM + MT) * 2.54
NUM_FS = 10.0 * FIGH_CM / 4.0             # 10 pt once the panel is 5 cm tall
K = {'fraction': H_CM / 9.71, 'scatter': H_CM / 9.82, 'metrics': H_CM / 9.30}
SPINE_LW, TICK_LEN, TICK_W = 0.5, 2.2, 0.5

# -- the wider panels (ablation, focal, training curves) ----------------------
FS_TICK, FS_LAB, FS_LEG = 19, 21, 15
OKABE = cfg.OKABE_ITO
C_CROP, C_NOCROP = COLORS['transformer'], COLORS['cnn']


def strip(ax, xticks=None, xlabels=None, ynums=True, yticks=None, ylabels=None):
    """The shared minimal chrome of the small panels."""
    yt = YTICKS if yticks is None else np.asarray(yticks)
    ax.set_yticks(yt)
    if ynums:
        ax.set_yticklabels([f'{t:.1f}' for t in yt] if ylabels is None else ylabels,
                           fontsize=NUM_FS, fontfamily=plots.PAPER_FONT)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='y', length=TICK_LEN, width=TICK_W, labelleft=ynums)
    if xticks is None:
        ax.set_xticks([])
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels([] if xlabels is None else xlabels,
                           fontsize=NUM_FS, fontfamily=plots.PAPER_FONT)
        ax.tick_params(axis='x', length=TICK_LEN, width=TICK_W,
                       labelbottom=xlabels is not None)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(SPINE_LW)


def save(fig, stem, ds):
    plt.rcParams['savefig.bbox'] = 'standard'
    return plots.save_figure(fig, cfg.figure_dir('paper') / f'{stem}_{ds.key}.png')


def save_tight(fig, stem, ds):
    plt.rcParams['savefig.bbox'] = 'tight'
    return plots.save_figure(fig, cfg.figure_dir('paper') / f'{stem}_{ds.key}.png')


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

def fraction_ev(ds, device, refresh=False):
    """Per-neuron EV of every fraction-study run, cached per dataset."""
    cache = cfg.cache_dir(ds) / 'fraction_ev.npz'
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as z:
            return ev.unflatten_fraction_results({k: z[k] for k in z.files})
    res = ev.fraction_metrics(CURVE_MODELS, metric='ev', ds=ds, device=device)
    if not res:
        return None
    cfg.ensure(cache.parent)
    np.savez(cache, **ev.flatten_fraction_results(res))
    print(f'  cached -> {cache}')
    return res


def ln_ev(ds, device):
    """(N,) EV of the LN model, averaged over its folds and per-neuron repeats."""
    rep = dat.test_repeats(ds)
    imgs = dat.images('test', ds)
    per_run = []
    for _fold, cells, conf in ckpt.load_ln_folds(ds, device):
        preds = ev.predict_ln(cells, imgs, int(conf['CROP_SIZE']), ds, device,
                              per_seed=True)
        per_run.extend(explained_variance_per_neuron(p, rep) for p in preds)
    return np.stack(per_run).mean(axis=0)


def ev_per_neuron(ds, device, refresh=False):
    """{model: (N,) EV} at the full training set, for the boxplot and scatter.

    Preferably from the fraction study at fraction 1.0, which averages the
    runs of every fold (and seed); a model with no sweep falls back to its
    canonical K-fold run, whose evaluation is likewise the mean over the folds.
    Those two sources are only comparable when the sweep was trained to the
    same budget as the canonical run, so where each model's numbers came from
    is reported, and a mixture is called out: a sweep cut short with --epochs
    would otherwise be plotted against fully trained models, significance
    stars and all, with nothing on the figure to say so.
    """
    res = fraction_ev(ds, device, refresh)
    out, source = {}, {}
    if res:
        for m in CURVE_MODELS:
            v = ev.per_neuron_at(res, m, 1.0, ds)
            if v is not None:
                out[m], source[m] = v, 'sweep'
    for m in CURVE_MODELS:                       # fall back to the canonical run
        if m not in out:
            e = ev.load_eval(m, ds)
            if e is not None:
                out[m], source[m] = e['explained_variance'], 'checkpoint'
    if ckpt.exists('LN', ds=ds):
        out['LN'], source['LN'] = ln_ev(ds, device), 'checkpoint'

    for m in ORDER:
        if m in out:
            print(f'  [source] {ABBR[m]:>5}: {source[m]:>10}  '
                  f'EV {float(np.mean(out[m])):.4f}')
    if len(set(source.values())) > 1:
        mixed = ', '.join(f'{ABBR[m]}={source[m]}' for m in ORDER if m in source)
        print(f'  [WARNING] this panel mixes sources ({mixed}). They are only '
              f'comparable if the sweep used the same training budget as the '
              f'canonical runs; finish the sweep, or read the panel with that '
              f'in mind.')
    return out


def lsta_corr_per_neuron(ds, device):
    """{model: (N,) LSTA correlation}, recomputed from the stored gradients.

    Recomputing rather than reading the stored scalar keeps every model on the
    current band limit, including the LN, whose maps are computed here. The
    correlation is computed per fold and averaged over the folds -- the same
    quantity `evaluate.py` stores -- from the per-fold gradient maps.
    """
    ref = dat.lsta_reference(ds)
    out = {}
    for m in CURVE_MODELS:
        per_fold = [e['lsta_model'] for e in ev.load_fold_evals(m, ds).values()
                    if 'lsta_model' in e]
        if not per_fold:
            continue
        out[m] = np.stack([
            L.corr_from_arrays(g, ref['lsta'], ref['ellipses'], g.shape[-1], ds=ds)
            for g in per_fold]).mean(axis=0)
    if ckpt.exists('LN', ds=ds):
        per_run = []
        for _fold, cells, conf in ckpt.load_ln_folds(ds, device):
            maps = L.ln_gradients(cells, dat.lsta_base_images(ds),
                                  int(conf['CROP_SIZE']), ds, device, per_seed=True)
            per_run.extend(
                L.corr_from_arrays(g, ref['lsta'], ref['ellipses'], g.shape[-1], ds=ds)
                for g in maps)
        out['LN'] = np.stack(per_run).mean(axis=0)
    return out


def ablation_store(ds):
    path = cfg.experiment_dir(ds) / 'ablation.json'
    if not path.exists():
        print(f'  [skip] {path} missing -- run scripts/ablation.py')
        return None
    return json.loads(path.read_text(encoding='utf-8'))


# -----------------------------------------------------------------------------
# 1. metrics_ev
# -----------------------------------------------------------------------------

def fig_metrics_ev(ds, device, data=None):
    data = data or ev_per_neuron(ds, device)
    order = [m for m in ORDER if m in data]
    if len(order) < 2:
        print('  [skip] metrics_ev: need at least two models')
        return
    box = [data[m] for m in order]
    km = K['metrics']

    fig, ax = plots.fixed_axes(W_METRICS, H_CM, mb=MB, mt=MT)
    plots.boxplot(ax, box, [COLORS[m] for m in order], width=0.62, points=False,
                  centre='median', lw=km)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(*YLIM)
    strip(ax)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([ABBR[m] for m in order], fontsize=NUM_FS,
                       fontfamily=plots.PAPER_FONT)
    ax.tick_params(axis='x', length=0, labelbottom=True)
    for lab, m in zip(ax.get_xticklabels(), order):
        lab.set_color(COLORS[m])
    if 'transformer' in order:
        plots.add_significance_stars(ax, box, order, 'transformer',
                                     fontsize=NUM_FS * 0.7)
    save(fig, 'metrics_ev', ds)


# -----------------------------------------------------------------------------
# 2. scatter_ev
# -----------------------------------------------------------------------------

def fig_scatter_ev(ds, device, data=None, reference='cnn'):
    data = data or ev_per_neuron(ds, device)
    if 'transformer' not in data or reference not in data:
        print(f'  [skip] scatter_ev: need transformer and {reference}')
        return
    ks = K['scatter']
    fig, ax = plots.fixed_axes(W_SCATTER, H_CM, mb=MB, mt=MT)
    ax.plot(YLIM, YLIM, '--', color='0.55', linewidth=1.4 * ks, zorder=1)
    ax.scatter(data[reference], data['transformer'], s=55 * ks * ks,
               color=COLORS[reference], edgecolor='black', linewidth=0.5 * ks,
               alpha=0.85, zorder=3)
    ax.set_xlim(*YLIM)
    ax.set_ylim(*YLIM)
    ticks = np.array([0.0, 0.5, 1.0])
    labels = [f'{t:.1f}' for t in ticks]
    strip(ax, xticks=ticks, xlabels=labels, yticks=ticks, ylabels=labels)
    better = int(np.sum(data['transformer'] > data[reference]))
    print(f'  CiRF-ViT above {ABBR[reference]} on {better}/{len(data["transformer"])} neurons')
    save(fig, f'scatter_ev_transformer_vs_{ABBR[reference]}', ds)


# -----------------------------------------------------------------------------
# 3. fraction_ev
# -----------------------------------------------------------------------------

def fig_fraction_ev(ds, device, results=None):
    results = results if results is not None else fraction_ev(ds, device)
    if not results:
        print('  [skip] fraction_ev -- run scripts/fraction_study.py')
        return
    kf = K['fraction']
    fig, ax = plots.fixed_axes(W_FRACTION, H_CM, ml=0.55, mr=0.14, mb=MB, mt=MT)
    present = []
    for m in CURVE_MODELS:
        xs, means, bands = ev.curve(results, m, ds)
        if not len(xs):
            continue
        present.append(m)
        ax.fill_between(xs, means - bands, means + bands, color=COLORS[m],
                        alpha=0.13, linewidth=0)
        ax.plot(xs, means, color=COLORS[m], linewidth=1.8 * kf,
                marker=cfg.MODEL_MARKERS[m], markersize=5 * kf,
                markeredgecolor='white', markeredgewidth=0.5 * kf)
    ax.set_xlim(0, 105)
    ax.set_ylim(*YLIM)
    xp = list(range(20, 101, 20))
    strip(ax, xticks=xp, xlabels=[str(p) for p in xp])
    for j, m in enumerate(present):
        ax.text(0.97, 0.06 + (len(present) - 1 - j) * 0.105, ABBR[m],
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=NUM_FS * 0.85, fontfamily=plots.PAPER_FONT,
                color=COLORS[m], zorder=6)
    save(fig, 'fraction_ev', ds)


# -----------------------------------------------------------------------------
# 4. metrics_lsta_corr
# -----------------------------------------------------------------------------

def fig_metrics_lsta_corr(ds, device):
    if not dat.meta(ds)['has_lsta']:
        print('  [skip] metrics_lsta_corr: no experimental LSTA in this dataset')
        return
    data = lsta_corr_per_neuron(ds, device)
    order = [m for m in ORDER if m in data]
    if len(order) < 2:
        print('  [skip] metrics_lsta_corr -- run scripts/evaluate.py first')
        return
    for m in order:
        print(f'  {cfg.MODEL_LABELS[m]:>12}  LSTA r = {data[m].mean():.4f}')
    box = [data[m] for m in order]
    km = K['metrics']

    fig, ax = plots.fixed_axes(W_METRICS, H_CM, mb=MB, mt=MT)
    plots.boxplot(ax, box, [COLORS[m] for m in order], width=0.62, points=False,
                  centre='mean', lw=km)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(*YLIM)
    strip(ax)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([ABBR[m] for m in order], fontsize=NUM_FS,
                       fontfamily=plots.PAPER_FONT)
    ax.tick_params(axis='x', length=0, labelbottom=True)
    for lab, m in zip(ax.get_xticklabels(), order):
        lab.set_color(COLORS[m])
    if 'transformer' in order:
        plots.add_significance_stars(ax, box, order, 'transformer',
                                     fontsize=NUM_FS * 0.7)
    save(fig, 'metrics_lsta_corr', ds)


# -----------------------------------------------------------------------------
# 5. ablation   6. focal_comparison
# -----------------------------------------------------------------------------

ABL_ORDER = ('no_crop', 'readout_crop', 'no_block', 'no_rope', 'no_attn', 'cnn_crop')
ABL_LABEL = {'no_crop': 'no crop', 'readout_crop': 'global attention,\ncrop in readout',
             'no_block': 'no transformer\nblock', 'no_rope': 'no RoPE',
             'no_attn': 'no attention', 'cnn_crop': 'CNN,\ncrop in readout'}
ABL_COLOR = {'no_crop': OKABE['vermillion'], 'readout_crop': OKABE['purple'],
             'no_block': OKABE['blue'], 'no_rope': OKABE['green'],
             'no_attn': OKABE['orange'], 'cnn_crop': OKABE['sky_blue']}


def _variant_ev(store, name):
    """(n_runs, N) EV of one ablation variant, or None."""
    entry = store.get(name)
    if not entry or not entry.get('ev'):
        return None
    return np.asarray(entry['ev'], dtype=float)


def reference_runs(results, model_type, ds):
    """(n_runs, N) EV of a model at the full training set, or None.

    From the data-efficiency sweep at 100 % when it has been run, else from
    the model's canonical folds (`scripts/evaluate.py`) -- the same
    trainings, so either source gives the same reference.
    """
    runs = ev.per_run_at(results, model_type, 1.0, ds) if results else None
    if runs is None:
        data = ev.load_eval(model_type, ds)
        if data is not None and 'explained_variance_folds' in data:
            runs = np.asarray(data['explained_variance_folds'], dtype=float)
    return runs


def fig_ablation(ds, device, reference_ev=None, cnn_ev=None):
    """One box per variant; dashed lines mark the full model and the CNN.

    The thick line in each box is the mean over neurons (the number printed
    under it). The CNN line is the reference of the `cnn_crop` control (the
    same CNN with the crop on its readout), as the full model is of the rest.
    """
    store = ablation_store(ds)
    if store is None:
        return
    present = [a for a in ABL_ORDER if _variant_ev(store, a) is not None]
    if not present:
        print('  [skip] ablation: no variants in ablation.json')
        return
    evs = {a: _variant_ev(store, a) for a in present}
    box = [evs[a].mean(axis=0) for a in present]

    plots.paper_style()
    fig, ax = plt.subplots(figsize=(max(6.8, 2.0 * len(present)), 3.8))
    plots.boxplot(ax, box, [ABL_COLOR[a] for a in present], centre='mean')

    if reference_ev is not None:
        ax.axhline(reference_ev, color='black', ls='--', lw=2.8, zorder=5,
                   label=f'full model (EV = {reference_ev:.3f})')
    if cnn_ev is not None and 'cnn_crop' in present:
        ax.axhline(cnn_ev, color=COLORS['cnn'], ls='--', lw=2.8, zorder=5,
                   label=f'CNN (EV = {cnn_ev:.3f})')
    if reference_ev is not None or cnn_ev is not None:
        ax.legend(fontsize=FS_LEG, frameon=False, loc='lower right')

    labels = []
    for a in present:
        per_run = evs[a].mean(axis=1)
        # every label takes three rows -- two for the name, one for the mean --
        # so the means line up whatever the length of the name
        name = ABL_LABEL[a] if '\n' in ABL_LABEL[a] else '\n' + ABL_LABEL[a]
        labels.append(f'{name}\nmean={per_run.mean():.3f}±{per_run.std():.3f}')
        print(f'  {a:12s}: EV = {per_run.mean():.4f} ± {per_run.std():.4f}')
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel('Explained variance', fontsize=FS_LAB)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    plots.style_axes(ax, tick_fs=FS_TICK)
    ax.tick_params(axis='x', labelsize=10.5)
    fig.tight_layout()
    save_tight(fig, 'ablation', ds)


FOC_ORDER = ('crop', 'focal', 'nocrop')
FOC_LABEL = {'crop': 'CiRF-ViT', 'focal': 'ViT-P', 'nocrop': 'no-crop'}
FOC_COLOR = {'crop': OKABE['vermillion'], 'focal': OKABE['blue'],
             'nocrop': OKABE['orange']}


def fig_focal_comparison(ds, device, crop_runs=None):
    store = ablation_store(ds)
    if store is None:
        return
    variants = {}
    if crop_runs is not None:
        variants['crop'] = crop_runs
    focal = _variant_ev(store, 'focal')
    nocrop = _variant_ev(store, 'no_crop')
    if focal is not None:
        variants['focal'] = focal
    if nocrop is not None:
        variants['nocrop'] = nocrop
    present = [v for v in FOC_ORDER if v in variants]
    if len(present) < 2:
        print('  [skip] focal_comparison: needs the focal and no-crop variants')
        return

    box = [variants[v].mean(axis=0) for v in present]
    plots.paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    plots.boxplot(ax, box, [FOC_COLOR[v] for v in present], centre='mean')

    labels = []
    for v in present:
        per_run = variants[v].mean(axis=1)
        tag = (f'{per_run.mean():.3f}±{per_run.std():.3f}' if len(per_run) > 1
               else f'{per_run.mean():.3f} (1 run)')
        labels.append(f'{FOC_LABEL[v]}\nmean={tag}')
        print(f'  {v:7s}: EV = {per_run.mean():.4f}')
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel('Explained variance', fontsize=FS_LAB)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    plots.style_axes(ax, tick_fs=FS_TICK)
    ax.tick_params(axis='x', labelsize=10.5)
    fig.tight_layout()
    save_tight(fig, 'focal_comparison', ds)


# -----------------------------------------------------------------------------
# 7. crop_vs_nocrop
# -----------------------------------------------------------------------------

def early_stop_epoch(val_loss, patience=30):
    best, best_ep, waited = float('inf'), 0, 0
    for i, v in enumerate(val_loss):
        if v < best:
            best, best_ep, waited = v, i, 0
        else:
            waited += 1
            if waited >= patience:
                break
    return best_ep


def fig_crop_vs_nocrop(ds, patience=30):
    nocrop_path = cfg.experiment_dir(ds) / 'history_nocrop.json'
    if not nocrop_path.exists():
        print('  [skip] crop_vs_nocrop -- run scripts/crop_vs_nocrop.py')
        return
    hist_nc = json.loads(nocrop_path.read_text(encoding='utf-8'))
    # the no-crop run was trained on one fold; compare it with the crop model
    # of that same fold, so both curves were stopped on the same stimuli
    fold = int(hist_nc.get('fold', 0))
    crop_path = cfg.fold_dir('transformer', fold, ds) / 'history.json'
    if not crop_path.exists():
        print(f'  [skip] crop_vs_nocrop -- fold {fold} of CiRF-ViT is missing '
              f'({crop_path})')
        return
    hist_crop = json.loads(crop_path.read_text(encoding='utf-8'))

    plots.paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for hist, col, name in ((hist_crop, C_CROP, 'crop'),
                            (hist_nc, C_NOCROP, 'no-crop')):
        epochs = np.arange(1, len(hist['train_loss']) + 1)
        ax.plot(epochs, hist['train_loss'], color=col, ls='-', lw=2.4,
                label=f'{name} train')
        ax.plot(epochs, hist['val_loss'], color=col, ls='--', lw=2.4,
                label=f'{name} val')
        i = early_stop_epoch(hist['val_loss'], patience)
        ax.plot(epochs[i], hist['val_loss'][i], marker='D', ms=9, mfc=col,
                mec='black', mew=0.9, ls='none', zorder=5)
    ax.set_xlabel('Epoch', fontsize=FS_LAB)
    ax.set_ylabel('Poisson NLL loss', fontsize=FS_LAB)
    ax.set_xlim(left=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    plots.style_axes(ax, tick_fs=FS_TICK)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker='D', ms=9, mfc='0.5', mec='black',
                          ls='none', label='early-stop pick'))
    fig.legend(handles=handles, fontsize=13, frameon=False, loc='lower center',
               ncol=3, columnspacing=1.4, handlelength=1.8)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    save_tight(fig, 'crop_vs_nocrop', ds)


# -----------------------------------------------------------------------------

FIGURES = ('metrics_ev', 'scatter_ev', 'fraction_ev', 'metrics_lsta_corr',
           'ablation', 'focal_comparison', 'crop_vs_nocrop')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(ap, default='both')
    add_device_arg(ap, default='cpu')
    ap.add_argument('--figures', nargs='+', default=list(FIGURES), choices=FIGURES)
    ap.add_argument('--refresh', action='store_true',
                    help='recompute the cached fraction-study metrics')
    ap.add_argument('--full-ev', type=float, default=None,
                    help='ablation: draw the full-model line at this EV instead '
                         'of the trained folds (e.g. a published value)')
    ap.add_argument('--cnn-ev', type=float, default=None,
                    help='ablation: the same for the CNN line')
    args = ap.parse_args()

    device = device_from(args, default='cpu')
    plots.paper_style()

    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        banner(f'paper figures | {ds.label}')

        results = fraction_ev(ds, device, args.refresh) \
            if {'fraction_ev', 'metrics_ev', 'scatter_ev', 'focal_comparison'} \
            & set(args.figures) else None
        data = ev_per_neuron(ds, device) \
            if {'metrics_ev', 'scatter_ev'} & set(args.figures) else None

        if 'metrics_ev' in args.figures:
            fig_metrics_ev(ds, device, data)
        if 'scatter_ev' in args.figures:
            fig_scatter_ev(ds, device, data)
        if 'fraction_ev' in args.figures:
            fig_fraction_ev(ds, device, results)
        if 'metrics_lsta_corr' in args.figures:
            fig_metrics_lsta_corr(ds, device)

        crop_runs = reference_runs(results, 'transformer', ds)
        cnn_runs = reference_runs(results, 'cnn', ds)
        ref_ev = float(crop_runs.mean()) if crop_runs is not None else None
        cnn_ev = float(cnn_runs.mean()) if cnn_runs is not None else None
        if 'ablation' in args.figures:
            fig_ablation(ds, device,
                         ref_ev if args.full_ev is None else args.full_ev,
                         cnn_ev if args.cnn_ev is None else args.cnn_ev)
        if 'focal_comparison' in args.figures:
            fig_focal_comparison(ds, device, crop_runs)
        if 'crop_vs_nocrop' in args.figures:
            fig_crop_vs_nocrop(ds)

    print(f'\nfigures in {cfg.figure_dir("paper")}')


if __name__ == '__main__':
    main()
