#!/usr/bin/env python
"""Example LSTA maps: one column per (neuron, stimulus), one row per model.

    python scripts/make_lsta_examples.py                       # paper's examples
    python scripts/make_lsta_examples.py --neuron 7 --stimuli 0 3
    python scripts/make_lsta_examples.py --data my.npz --neuron 0

This is the content of the paper's LSTA figure panel: for a given neuron and two
stimuli, the stimulus itself with the receptive-field outline, the measured
LSTA, and every model's input gradient -- all band-limited exactly as the metric
consumes them, so a panel shows what the numbers actually score.

Pick stimuli on which the experimental map has *opposite* polarity for the same
neuron: that inversion is the context dependence a linear model cannot produce
by construction, and it is what the panel exists to show. The defaults are the
neurons and stimuli used in the paper (mouse neuron 21, stimuli 16 and 0;
salamander neuron 5, stimuli 30 and 0).

The display crop is tighter than the metric's window (--padding, default 4
against 8) and squared around the receptive field, for legibility only.
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import (add_dataset_arg, add_device_arg, banner, cfg,
                     datasets_from, device_from, require_lsta)
from cirf import checkpoints as ckpt
from cirf import data as dat
from cirf import evaluate as ev
from cirf import lsta as L
from cirf import plots

import matplotlib.pyplot as plt

#: the paper's examples: (neuron, [stimulus ids, as in cirf.data.STIMULUS_IDS])
PAPER_EXAMPLES = {'mice': (21, [16, 0]), 'salamander': (5, [30, 0])}

ROWS = ('experiment', 'transformer', 'LN', 'cnn', 'GP')
ROW_LABEL = {'experiment': 'Exp', 'transformer': 'CiRF', 'LN': 'LN',
             'cnn': 'CNN', 'GP': 'GP'}
ROW_COLOR = dict(cfg.PAPER_COLORS, experiment='black')
GREEN = '#39FF14'


def square_box(ex, ey, padding, size):
    """A square display window around the RF outline, clipped to the frame."""
    x0, x1 = max(0.0, ex.min() - padding), min(float(size), ex.max() + padding)
    y0, y1 = max(0.0, ey.min() - padding), min(float(size), ey.max() + padding)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = cx - side / 2, cx + side / 2
    y0, y1 = cy - side / 2, cy + side / 2
    if x0 < 0:
        x1, x0 = x1 - x0, 0.0
    if y0 < 0:
        y1, y0 = y1 - y0, 0.0
    if x1 > size:
        x0, x1 = x0 - (x1 - size), float(size)
    if y1 > size:
        y0, y1 = y0 - (y1 - size), float(size)
    return (int(round(max(0.0, x0))), int(round(min(float(size), x1))),
            int(round(max(0.0, y0))), int(round(min(float(size), y1))))


def collect(ds, neuron, stim_ids, device):
    """{row: [(map, vmax) per stimulus]} plus the stimuli and the outline."""
    ref = dat.lsta_reference(ds)
    base_imgs = dat.lsta_base_images(ds)
    ids = list(np.asarray(ref['image_indices']))
    try:
        sel = [ids.index(s) for s in stim_ids]
    except ValueError:
        raise SystemExit(f'stimuli {stim_ids} not among the LSTA stimuli {ids}')

    maps = {'experiment': np.stack([ref['lsta'][neuron, j] for j in sel])}
    for m in ('transformer', 'cnn', 'GP'):
        e = ev.load_eval(m, ds)
        if e is None or 'lsta_model' not in e:
            print(f'  [skip row] {m}: no gradients in its eval_data.npz')
            continue
        maps[m] = np.stack([e['lsta_model'][neuron, j] for j in sel])
    if ckpt.exists('LN', ds=ds):
        # mean map over the folds, like the stored `lsta_model` of the others
        ln = np.mean([
            L.ln_gradients([cells[neuron]], base_imgs, int(conf['CROP_SIZE']),
                           ds, device)[0]
            for _fold, cells, conf in ckpt.load_ln_folds(ds, device)], axis=0)
        maps['LN'] = np.stack([ln[j] for j in sel])

    processed = {r: [L.prepare_panel(a, ds=ds) for a in arr]
                 for r, arr in maps.items()}
    return processed, [base_imgs[j] for j in sel]


def make_figure(ds, neuron, stim_ids, device, padding):
    processed, stimuli = collect(ds, neuron, stim_ids, device)
    ref = dat.lsta_reference(ds)
    P = dat.img_size(ds)
    scale = P / dat.ellipse_frame(ds)
    ex = ref['ellipses'][neuron, 0, :] * scale
    ey = ref['ellipses'][neuron, 1, :] * scale
    x0, x1, y0, y1 = square_box(ex, ey, padding, P)
    extent = [x0, x1, y1, y0]

    rows = ['stimulus'] + [r for r in ROWS if r in processed]
    n_col = len(stim_ids)
    plots.paper_style(font_size=11)
    fig, axes = plt.subplots(len(rows), n_col,
                             figsize=(n_col * 1.7, len(rows) * 1.8), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_col):
            ax = axes[r, c]
            if row == 'stimulus':
                ax.imshow(stimuli[c][y0:y1, x0:x1], cmap='gray', origin='upper',
                          extent=extent)
                ax.plot(ex, ey, color=GREEN, lw=1.4)
                ax.set_title(f'stimulus {stim_ids[c]}', fontsize=10)
            else:
                pmap, vmax = processed[row][c]
                ax.imshow(pmap[y0:y1, x0:x1], cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          origin='upper', extent=extent, interpolation='bicubic')
                ax.plot(ex, ey, color='k', lw=1.0)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if c == 0:
                label = 'Img' if row == 'stimulus' else ROW_LABEL[row]
                ax.set_ylabel(label, fontsize=12, fontweight='bold',
                              color=ROW_COLOR.get(row, 'black'))
    fig.suptitle(f'{ds.label} -- neuron {neuron}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plots.save_figure(fig, cfg.figure_dir('paper') /
                      f'lsta_examples_{ds.key}_neuron{neuron:02d}.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(ap, default='both')
    add_device_arg(ap, default='cpu')
    ap.add_argument('--neuron', type=int, default=None,
                    help='neuron index (default: the paper example)')
    ap.add_argument('--stimuli', type=int, nargs='+', default=None,
                    help='stimulus ids, as in cirf.data.STIMULUS_IDS')
    ap.add_argument('--padding', type=int, default=4,
                    help='display padding around the RF outline (metric uses 8)')
    args = ap.parse_args()

    device = device_from(args, default='cpu')
    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        if not require_lsta(ds):
            continue
        neuron, stimuli = PAPER_EXAMPLES.get(
            ds.key, (0, list(np.asarray(dat.lsta_reference(ds)['image_indices'])[:2])))
        neuron = args.neuron if args.neuron is not None else neuron
        stimuli = args.stimuli or stimuli
        banner(f'LSTA examples | {ds.label} | neuron {neuron} | stimuli {stimuli}')
        make_figure(ds, neuron, stimuli, device, args.padding)


if __name__ == '__main__':
    main()
