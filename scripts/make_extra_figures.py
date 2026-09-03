#!/usr/bin/env python
"""Supplementary figures: everything that is not in the paper.

    python scripts/make_extra_figures.py                       # all of them
    python scripts/make_extra_figures.py --figures lsta_grids --dataset mouse
    python scripts/make_extra_figures.py --data my.npz

Written to results/figures/extra/, in plain matplotlib style (these are for
reading, not for typesetting):

    metrics_all        EV, adjusted R^2 and LSTA correlation side by side
    metrics_clean      EV and adjusted R^2 averaged over the fraction-study runs
    scatter_ev_gp      per-neuron EV, CiRF-ViT against the GP
    fraction_adj_r2    the data-efficiency curve in adjusted R^2, plus the
                       per-neuron distribution at the full training set
    training_curves    training and validation loss of every model
    response_stats     firing-rate distribution per neuron, and the predicted
                       against measured response of each model
    response_rank      per neuron, the test images sorted by measured response
                       with every model's prediction on top
    lsta_grids         one panel per neuron: stimulus, experimental LSTA and
                       every model's gradient, band-limited exactly as the
                       metric consumes them

Inputs are the eval_data.npz files (scripts/evaluate.py) and, for the fraction
figures, the sweep (scripts/fraction_study.py). Missing inputs skip a figure
with a message rather than failing.
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import (add_dataset_arg, add_device_arg, banner, cfg,
                     datasets_from, device_from)
from cirf import checkpoints as ckpt
from cirf import data as dat
from cirf import evaluate as ev
from cirf import lsta as L
from cirf import plots
from cirf.metrics import METRIC_TITLES, NPZ_KEY

import matplotlib.pyplot as plt

ORDER = ('LN', 'GP', 'cnn', 'transformer')
CURVE_MODELS = ('GP', 'cnn', 'transformer')
COLORS = {'transformer': 'C0', 'cnn': 'C1', 'GP': 'C2', 'LN': 'C3'}
LABELS = cfg.MODEL_LABELS
MARKERS = cfg.MODEL_MARKERS

METRIC_SPECS = (('ev', (0.0, 1.0)), ('adj_r2', (0.0, 1.0)),
                ('lsta_corr', (-0.3, 1.0)))


def out_path(stem, ds):
    return cfg.figure_dir('extra') / f'{stem}_{ds.key}.png'


def loaded_evals(ds):
    return {m: e for m in ORDER if (e := ev.load_eval(m, ds)) is not None}


# -----------------------------------------------------------------------------
# metrics_all / metrics_clean
# -----------------------------------------------------------------------------

def _metric_panel(ax, key, ylim, present, data):
    values = [np.asarray(data[m][NPZ_KEY[key]], dtype=float) for m in present]
    plots.boxplot(ax, values, [COLORS[m] for m in present], centre='median',
                  point_size=13)
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([f'{LABELS[m]}\nmean={v.mean():.3f}'
                        for m, v in zip(present, values)])
    ax.set_ylabel(METRIC_TITLES[key])
    ax.set_ylim(ylim[0] - 0.05, ylim[1] + 0.05)
    ax.set_title(METRIC_TITLES[key])
    plots.despine(ax)


def fig_metrics_all(ds, device):
    data = loaded_evals(ds)
    if not data:
        print('  [skip] metrics_all -- run scripts/evaluate.py')
        return
    present = [m for m in ORDER if m in data]
    specs = [(k, yl) for k, yl in METRIC_SPECS
             if any(np.any(data[m][NPZ_KEY[k]]) for m in present)]
    fig, axes = plt.subplots(1, len(specs), figsize=(4.6 * len(specs), 4.2),
                             squeeze=False)
    for ax, (key, ylim) in zip(axes[0], specs):
        _metric_panel(ax, key, ylim, present, data)
    fig.tight_layout()
    plots.save_figure(fig, out_path('metrics_all', ds))


def fig_metrics_clean(ds, device):
    """EV and adjusted R^2 per neuron, averaged over the fraction-study runs."""
    for metric in ('ev', 'adj_r2'):
        res = ev.fraction_metrics(CURVE_MODELS, fractions=[1.0], metric=metric,
                                  ds=ds, device=device, verbose=False)
        data = {}
        for m in CURVE_MODELS:
            v = ev.per_neuron_at(res, m, 1.0, ds)
            if v is not None:
                data[m] = v
        if ckpt.exists('LN', ds=ds):
            from cirf.metrics import METRIC_FNS
            rep = dat.test_repeats(ds)
            per_run = []
            for _fold, cells, conf in ckpt.load_ln_folds(ds, device):
                preds = ev.predict_ln(cells, dat.images('test', ds),
                                      int(conf['CROP_SIZE']), ds, device,
                                      per_seed=True)
                per_run.extend(METRIC_FNS[metric](p, rep) for p in preds)
            data['LN'] = np.stack(per_run).mean(0)
        present = [m for m in ORDER if m in data]
        if not present:
            print(f'  [skip] metrics_{metric}_clean -- run scripts/fraction_study.py')
            continue
        box = [data[m] for m in present]
        fig, ax = plt.subplots()
        plots.boxplot(ax, box, [COLORS[m] for m in present], centre='mean',
                      points=False, width=0.6)
        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels([LABELS[m] for m in present])
        ax.set_ylabel(f'{METRIC_TITLES[metric]} (mean over runs)')
        ax.set_title(METRIC_TITLES[metric])
        ax.set_ylim(-0.02, 1.05)
        fig.tight_layout()
        plots.save_figure(fig, out_path(f'metrics_{metric}_clean', ds))
        print('    ' + ', '.join(f'{m}={np.mean(data[m]):.3f}' for m in present))


# -----------------------------------------------------------------------------
# scatter / fraction curves
# -----------------------------------------------------------------------------

def fig_scatter_ev_gp(ds, device):
    res = ev.fraction_metrics(CURVE_MODELS, fractions=[1.0], metric='ev',
                              ds=ds, device=device, verbose=False)
    x = ev.per_neuron_at(res, 'GP', 1.0, ds)
    y = ev.per_neuron_at(res, 'transformer', 1.0, ds)
    if x is None or y is None:
        e_gp, e_tr = ev.load_eval('GP', ds), ev.load_eval('transformer', ds)
        if e_gp is None or e_tr is None:
            print('  [skip] scatter_ev_gp -- needs the GP and CiRF-ViT evaluations')
            return
        x, y = e_gp['explained_variance'], e_tr['explained_variance']
    lim = (0.0, 1.1)
    fig, ax = plt.subplots()
    ax.plot(lim, lim, '--', color='0.55', zorder=1)
    ax.scatter(x, y, color=COLORS['GP'], edgecolor='black', linewidth=0.5,
               alpha=0.85, zorder=3)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('GP explained variance')
    ax.set_ylabel('CiRF-ViT explained variance')
    ax.set_title(f'Per-neuron EV -- CiRF-ViT vs GP ({ds.key})')
    fig.tight_layout()
    plots.save_figure(fig, out_path('scatter_ev_transformer_vs_GP', ds))
    print(f'    CiRF-ViT above the GP on {int(np.sum(y > x))}/{x.size} neurons')


def fig_fraction_adj_r2(ds, device, refresh=False):
    cache = cfg.cache_dir(ds) / 'fraction_adj_r2.npz'
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as z:
            results = ev.unflatten_fraction_results({k: z[k] for k in z.files})
    else:
        results = ev.fraction_metrics(CURVE_MODELS, metric='adj_r2', ds=ds,
                                      device=device)
        if not results:
            print('  [skip] fraction_adj_r2 -- run scripts/fraction_study.py')
            return
        cfg.ensure(cache.parent)
        np.savez(cache, **ev.flatten_fraction_results(results))

    fig, ax = plt.subplots()
    for m in CURVE_MODELS:
        xs, means, bands = ev.curve(results, m, ds)
        if not len(xs):
            continue
        ax.fill_between(xs, means - bands, means + bands, color=COLORS[m],
                        alpha=0.15, linewidth=0)
        ax.plot(xs, means, color=COLORS[m], marker=MARKERS[m], label=LABELS[m])
    ax.set_xlabel('Training data (%)')
    ax.set_ylabel('Adjusted R²')
    ax.set_title(f'Data efficiency -- adjusted R² ({ds.key})')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    plots.save_figure(fig, out_path('fraction_adj_r2', ds))

    box, present = [], []
    for m in CURVE_MODELS:
        v = ev.per_neuron_at(results, m, 1.0, ds)
        if v is not None:
            box.append(v)
            present.append(m)
    if present:
        fig, ax = plt.subplots()
        plots.boxplot(ax, box, [COLORS[m] for m in present], centre='median',
                      point_size=14)
        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels([LABELS[m] for m in present])
        ax.set_ylabel('Adjusted R² (full training set)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'Adjusted R² at 100% of the data ({ds.key})')
        fig.tight_layout()
        plots.save_figure(fig, out_path('boxplot_100pct', ds))


# -----------------------------------------------------------------------------
# training curves / responses
# -----------------------------------------------------------------------------

def fig_training_curves(ds, device):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    drawn = False
    for m in ('cnn', 'transformer'):
        e = ev.load_eval(m, ds)
        if e is None or not len(e['train_loss']):
            continue
        drawn = True
        tr, vl = e['train_loss'], e['val_loss']
        ax.plot(np.arange(1, len(tr) + 1), tr, color=COLORS[m], ls='-',
                label=f'{LABELS[m]} (train)')
        ax.plot(np.arange(1, len(vl) + 1), vl, color=COLORS[m], ls='--',
                label=f'{LABELS[m]} (val)')
    if not drawn:
        print('  [skip] training_curves -- run scripts/evaluate.py')
        plt.close(fig)
        return
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Poisson loss')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    plots.despine(ax)
    fig.tight_layout()
    plots.save_figure(fig, out_path('training_curves', ds))


def fig_response_stats(ds, device):
    """Firing-rate distribution per neuron, and predicted vs measured rates."""
    d = dat.load(ds)
    pooled = np.concatenate([
        np.asarray(d['responses_pool'], np.float32),
        np.asarray(d['responses_test'], np.float32).reshape(-1, dat.n_cells(ds)),
    ], axis=0) / cfg.STIM_DURATION_S
    N = pooled.shape[1]

    fig, ax = plt.subplots(figsize=(max(12, N * 0.30), 5.0))
    ax.boxplot([pooled[:, i] for i in range(N)], positions=np.arange(N),
               widths=0.72, patch_artist=True, showfliers=False, manage_ticks=False)
    ax.set_xlim(-0.7, N - 0.3)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'Response distribution per neuron ({ds.key}, n={N})')
    fig.tight_layout()
    plots.save_figure(fig, out_path('response_boxplot', ds))

    for m, e in loaded_evals(ds).items():
        x = np.asarray(e['y_true_mean'], dtype=float).ravel()
        y = np.asarray(e['y_pred'], dtype=float).ravel()
        lim = (0.0, float(max(x.max(), y.max()) * 1.05))
        fig, ax = plt.subplots()
        ax.plot(lim, lim, '--', color='0.55', zorder=1)
        ax.scatter(x, y, s=14, color=COLORS[m], alpha=0.4, linewidth=0, zorder=3)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Measured response')
        ax.set_ylabel('Predicted response')
        ax.set_title(f'{LABELS[m]} -- predicted vs measured ({ds.key})')
        fig.tight_layout()
        plots.save_figure(fig, out_path(f'resp_true_vs_pred_{m}', ds))


def fig_response_rank(ds, device):
    data = loaded_evals(ds)
    if not data:
        print('  [skip] response_rank -- run scripts/evaluate.py')
        return
    ref = next(iter(data.values()))
    y_true = np.asarray(ref['test_rep']).mean(axis=1)
    N = y_true.shape[1]
    cols = 7
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.5), squeeze=False)
    handles_labels = None
    for i in range(N):
        ax = axes[i // cols, i % cols]
        order = np.argsort(y_true[:, i])
        ax.plot(y_true[order, i], color='black', linewidth=1.3, label='Experiment',
                zorder=4)
        for m, e in data.items():
            ax.plot(np.asarray(e['y_pred'])[order, i], color=COLORS[m],
                    linewidth=1.0, alpha=0.9, label=LABELS[m])
        ax.set_title(f'neuron {i:02d}', fontsize=6.5, pad=2)
        ax.tick_params(axis='both', labelsize=5.5, length=2)
        plots.despine(ax)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
    for j in range(N, rows * cols):
        axes[j // cols, j % cols].axis('off')
    if handles_labels:
        fig.legend(*handles_labels, loc='lower center', ncol=len(data) + 1,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.supxlabel('Test image (sorted by measured response)', fontsize=9, y=0.015)
    fig.supylabel('Response (a.u.)', fontsize=9, x=0.004)
    fig.tight_layout(rect=(0.015, 0.03, 1, 1))
    plots.save_figure(fig, out_path('response_rank', ds))


# -----------------------------------------------------------------------------
# LSTA grids
# -----------------------------------------------------------------------------

def fig_lsta_grids(ds, device, neurons=None):
    if not dat.meta(ds)['has_lsta']:
        print('  [skip] lsta_grids: no experimental LSTA in this dataset')
        return
    ref = dat.lsta_reference(ds)
    exp_lsta, ellipses = ref['lsta'], ref['ellipses']
    base_imgs = dat.lsta_base_images(ds)
    P = dat.img_size(ds)
    scale = P / dat.ellipse_frame(ds)

    maps = {}
    for m in ('GP', 'cnn', 'transformer'):
        e = ev.load_eval(m, ds)
        if e is not None and 'lsta_model' in e:
            maps[m] = e['lsta_model']
    if ckpt.exists('LN', ds=ds):
        # mean map over the folds, like the stored `lsta_model` of the others
        maps['LN'] = np.mean([
            L.ln_gradients(cells, base_imgs, int(conf['CROP_SIZE']), ds, device)
            for _fold, cells, conf in ckpt.load_ln_folds(ds, device)], axis=0)
    if not maps:
        print('  [skip] lsta_grids -- run scripts/evaluate.py')
        return

    models = [m for m in ORDER if m in maps]
    rows = ['stimulus', 'experiment'] + models
    n_img = exp_lsta.shape[1]
    out_dir = cfg.ensure(cfg.figure_dir('extra') / f'lsta_grids_{ds.key}')
    neurons = range(dat.n_cells(ds)) if neurons is None else neurons
    print(f'  {len(list(neurons)) if not isinstance(neurons, range) else len(neurons)} '
          f'neurons x {len(rows)} rows -> {out_dir}')

    for n in neurons:
        ex, ey = ellipses[n, 0, :] * scale, ellipses[n, 1, :] * scale
        x0 = int(max(0, ex.min() - L.PADDING))
        x1 = int(min(P, ex.max() + L.PADDING))
        y0 = int(max(0, ey.min() - L.PADDING))
        y1 = int(min(P, ey.max() + L.PADDING))
        extent = [x0, x1, y1, y0]

        panels = {('experiment', j): L.prepare_panel(exp_lsta[n, j], ds=ds)
                  for j in range(n_img)}
        for m in models:
            for j in range(n_img):
                panels[(m, j)] = L.prepare_panel(maps[m][n, j], ds=ds)

        fig, axes = plt.subplots(len(rows), n_img,
                                 figsize=(n_img * 1.6, len(rows) * 1.7), squeeze=False)
        for r, rid in enumerate(rows):
            label = ('Stimulus' if rid == 'stimulus' else
                     'Experiment' if rid == 'experiment' else LABELS[rid])
            color = 'black' if rid in ('stimulus', 'experiment') else COLORS[rid]
            for c in range(n_img):
                ax = axes[r, c]
                if rid == 'stimulus':
                    ax.imshow(base_imgs[c][y0:y1, x0:x1], cmap='gray',
                              origin='upper', extent=extent)
                    ax.plot(ex, ey, color='#39FF14', lw=1.2)
                else:
                    pmap, vmax = panels[(rid, c)]
                    ax.imshow(pmap[y0:y1, x0:x1], cmap='RdBu_r', vmin=-vmax,
                              vmax=vmax, origin='upper', extent=extent,
                              interpolation='bicubic')
                    ax.plot(ex, ey, color='k', lw=1.0)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(label, fontweight='bold', color=color)
                if r == 0:
                    ax.set_title(f'stimulus {int(ref["image_indices"][c])}')
        fig.suptitle(f'Local STA -- neuron {n:02d}   '
                     f'(red +, blue -; outline = receptive field)')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plots.save_figure(fig, out_dir / f'lsta_neuron_{n:02d}.png')


# -----------------------------------------------------------------------------

FIGURES = ('metrics_all', 'metrics_clean', 'scatter_ev_gp', 'fraction_adj_r2',
           'training_curves', 'response_stats', 'response_rank', 'lsta_grids')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(ap, default='both')
    add_device_arg(ap, default='cpu')
    ap.add_argument('--figures', nargs='+', default=list(FIGURES), choices=FIGURES)
    ap.add_argument('--neurons', type=int, nargs='+', default=None,
                    help='lsta_grids: only these neurons')
    ap.add_argument('--refresh', action='store_true')
    args = ap.parse_args()

    device = device_from(args, default='cpu')
    plots.default_style()

    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        banner(f'supplementary figures | {ds.label}')
        for name in args.figures:
            print(f'\n-- {name} --')
            if name == 'metrics_all':
                fig_metrics_all(ds, device)
            elif name == 'metrics_clean':
                fig_metrics_clean(ds, device)
            elif name == 'scatter_ev_gp':
                fig_scatter_ev_gp(ds, device)
            elif name == 'fraction_adj_r2':
                fig_fraction_adj_r2(ds, device, args.refresh)
            elif name == 'training_curves':
                fig_training_curves(ds, device)
            elif name == 'response_stats':
                fig_response_stats(ds, device)
            elif name == 'response_rank':
                fig_response_rank(ds, device)
            elif name == 'lsta_grids':
                fig_lsta_grids(ds, device, args.neurons)

    print(f'\nfigures in {cfg.figure_dir("extra")}')


if __name__ == '__main__':
    main()
