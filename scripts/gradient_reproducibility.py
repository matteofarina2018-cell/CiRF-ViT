#!/usr/bin/env python
"""Is the fine structure in a model's gradients real, or training-run noise?

The LSTA benchmark shows that CiRF-ViT recovers the right polarity but that its
gradients are much rougher than the measured maps. Two readings compete:

  (A) missing smoothness prior -- nothing in the crop or the attention enforces
      smoothness, so the fine detail is noise, and it should NOT repeat across
      independently trained models;
  (B) systematic structure -- the model consistently learns a sensitivity the
      experiment cannot resolve, and the fine detail SHOULD repeat.

The two are separated by training-run resampling. The runs of the fraction study
at fraction 1.0 all see the identical training set (no subsetting happens at
that fraction) and differ only in initialisation, dropout and batch order, so
anything common to all of them comes from the data and the architecture.

Measured per neuron, inside the same receptive-field box the published metric
uses:

    seed vs experiment   reproduces the published LSTA correlation
    seed vs seed         agreement between two independently trained runs
    *_hp                 the same, on the high-pass residual only -- the fine
                         structure the question is actually about
    null_hp              high-pass agreement between *different* neurons: the
                         floor a meaningless comparison would give

Two controls guard the seed-vs-seed number: the different-neuron null above,
and `--interior`, which drops the crop-boundary annulus (the crop radius is a
fixed hyper-parameter shared by every run, so a discontinuity there would agree
across runs for free).

    python scripts/gradient_reproducibility.py --dataset both
    python scripts/gradient_reproducibility.py --model cnn --sigma 2
    python scripts/gradient_reproducibility.py --from-cache        # figures only
    python scripts/gradient_reproducibility.py --tables            # all models

Needs the fraction study at fraction 1.0 (scripts/fraction_study.py). Costs a
few minutes per dataset on CPU; results are cached under
results/<dataset>/cache/.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from _common import (add_dataset_arg, add_device_arg, banner, cfg,
                     datasets_from, device_from, require_lsta)
from cirf import data as dat
from cirf import lsta as L
from cirf import plots

SIGMA_SWEEP = (1.0, 2.0, 3.0, 5.0)
INTERIOR_FRAC = 0.7           # keep pixels within this share of the crop radius
TABLE_SIGMAS = (0, 0.5, 1.0, 1.5, 2.0, 2.2, 3.0, 5.0)

CACHE_NAME = 'gradient_reproducibility.npz'
FIG_COLORS = {'seedseed': '#0F5450', 'seedexp': '#C83C37', 'null': '#999999'}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def interior_masks(ds, frac=INTERIOR_FRAC, model_type='transformer'):
    """(N, P, P) masks keeping only the interior of each neuron's crop.

    Falls back to all-True when the radii are not recorded (any model without a
    crop), which makes the control a no-op rather than an error.
    """
    P = dat.img_size(ds)
    cx, cy = dat.centers(ds)
    radii = None
    for fold in cfg.FOLDS:
        for seed in cfg.FRAC_SEEDS:
            path = cfg.fraction_dir(model_type, 1.0, fold, seed, ds) / 'config.json'
            if path.exists():
                conf = json.loads(path.read_text(encoding='utf-8'))
                radii = conf.get('cfg', conf).get('NEURON_RADII')
                break
        if radii is not None:
            break
    if radii is None or len(radii) != len(cx):
        print('  ! no per-neuron radii recorded; interior control disabled')
        return np.ones((len(cx), P, P), dtype=bool)
    yy, xx = np.mgrid[0:P, 0:P]
    return np.stack([((yy - cy[n]) ** 2 + (xx - cx[n]) ** 2)
                     <= (frac * float(radii[n])) ** 2 for n in range(len(cx))])


def analyse(grads, exp_frame, boxes, sigma, masks=None):
    """Every correlation family, per neuron. See the module docstring."""
    S, N, I = grads.shape[:3]
    pairs = list(itertools.combinations(range(S), 2))
    grads_hp = L.highpass(grads, sigma)
    exp_hp = L.highpass(exp_frame, sigma)

    shapes = {'seedseed_pub': (len(pairs), N), 'seedseed_raw': (len(pairs), N),
              'seedseed_hp': (len(pairs), N), 'null_hp': (len(pairs), N),
              'seedexp_pub': (S, N), 'seedexp_raw': (S, N), 'seedexp_hp': (S, N)}
    out = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}

    rng = np.random.default_rng(0)
    partner = np.array([rng.choice([m for m in range(N) if m != n]) for n in range(N)])

    for n in range(N):
        y0, y1, x0, x1 = boxes[n]
        if y1 <= y0 or x1 <= x0:
            continue
        cut = (slice(y0, y1), slice(x0, x1))
        sel = masks[n][cut] if masks is not None else None
        if sel is not None and sel.sum() < 16:
            continue
        take = (lambda a: a[..., sel]) if sel is not None else (lambda a: a)

        pub = take(np.stack([[L.postprocess(grads[s, n, i])[0][cut]
                              for i in range(I)] for s in range(S)]))
        pub_e = take(np.stack([L.postprocess(exp_frame[n, i])[0][cut] for i in range(I)]))
        raw = take(grads[:, n][(slice(None), slice(None)) + cut])
        hp = take(grads_hp[:, n][(slice(None), slice(None)) + cut])
        raw_e = take(exp_frame[n][(slice(None),) + cut])
        hp_e = take(exp_hp[n][(slice(None),) + cut])
        hp_p = take(grads_hp[:, partner[n]][(slice(None), slice(None)) + cut])

        for s in range(S):
            out['seedexp_pub'][s, n] = np.mean([L.correlation(pub[s, i], pub_e[i]) for i in range(I)])
            out['seedexp_raw'][s, n] = np.mean([L.correlation(raw[s, i], raw_e[i]) for i in range(I)])
            out['seedexp_hp'][s, n] = np.mean([L.correlation(hp[s, i], hp_e[i]) for i in range(I)])
        for p, (a, b) in enumerate(pairs):
            out['seedseed_pub'][p, n] = np.mean([L.correlation(pub[a, i], pub[b, i]) for i in range(I)])
            out['seedseed_raw'][p, n] = np.mean([L.correlation(raw[a, i], raw[b, i]) for i in range(I)])
            out['seedseed_hp'][p, n] = np.mean([L.correlation(hp[a, i], hp[b, i]) for i in range(I)])
            out['null_hp'][p, n] = np.mean([L.correlation(hp[a, i], hp_p[b, i]) for i in range(I)])
    return out


def per_neuron(res):
    """Collapse the run / run-pair axis, leaving one value per neuron."""
    return {k: v.mean(axis=0) for k, v in res.items()}


def experimental_frame(ds):
    """(N, n_img, P, P) experimental LSTA resampled to the analysis frame."""
    ref = dat.lsta_reference(ds)
    P = dat.img_size(ds)
    lsta = ref['lsta']
    return np.stack([[L.resize_2d(lsta[n, i], P) for i in range(lsta.shape[1])]
                     for n in range(lsta.shape[0])]).astype(np.float32)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

ROWS = [
    ('published pipeline (signed-square + 20% threshold)', None),
    ('  model vs experiment  (= published LSTA correlation)', 'seedexp_pub'),
    ('  model vs model', 'seedseed_pub'),
    ('raw gradients', None),
    ('  model vs experiment', 'seedexp_raw'),
    ('  model vs model', 'seedseed_raw'),
    ('high-pass residual (fine structure only)', None),
    ('  model vs experiment', 'seedexp_hp'),
    ('  model vs model', 'seedseed_hp'),
    ('  model vs model, different neurons (null)', 'null_hp'),
]


def print_block(title, pn):
    n = len(next(iter(pn.values())))
    print('\n' + '=' * 74)
    print(f'{title}   (mean +/- sd over {n} neurons)')
    print('=' * 74)
    for label, key in ROWS:
        if key is None:
            print(f'\n{label}')
        else:
            print(f'{label:<52} {pn[key].mean():+.3f} +/- {pn[key].std():.3f}')


def print_sweep(sweep, label):
    print('\n' + '=' * 74)
    print(f'Sigma sweep -- high-pass residual, {label}')
    print('=' * 74)
    print(f'{"sigma":>7} {"model-model":>13} {"model-exp":>12} {"null":>12}')
    for sigma in sorted(sweep):
        pn = sweep[sigma]
        print(f'{sigma:>7.1f} {pn["seedseed_hp"].mean():>13.3f} '
              f'{pn["seedexp_hp"].mean():>12.3f} {pn["null_hp"].mean():>12.3f}')


def make_figure(pn, sweep_full, sweep_int, sigma, n_runs, ds, model_type):
    plots.default_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    def box(ax, groups, title):
        ax.boxplot([pn[k] for _, k, _ in groups],
                   tick_labels=[lbl for lbl, _, _ in groups], widths=0.5)
        for i, (_, k, c) in enumerate(groups, start=1):
            ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(pn[k])),
                       pn[k], s=10, alpha=0.5, color=c, zorder=3)
        ax.set_ylabel('correlation')
        ax.set_title(title)
        ax.axhline(0, color='0.7', lw=0.8, zorder=0)

    box(axes[0], [('model vs\nexperiment', 'seedexp_pub', FIG_COLORS['seedexp']),
                  ('model vs\nmodel', 'seedseed_pub', FIG_COLORS['seedseed'])],
        f'Published pipeline\n({n_runs} independent training runs)')
    box(axes[1], [('model vs\nexperiment', 'seedexp_hp', FIG_COLORS['seedexp']),
                  ('model vs\nmodel', 'seedseed_hp', FIG_COLORS['seedseed']),
                  ('different\nneurons', 'null_hp', FIG_COLORS['null'])],
        f'High-pass residual only\n(Gaussian sigma = {sigma:g} px)')

    ax = axes[2]
    for sweep, style, tag in ((sweep_full, '-', 'full RF box'),
                              (sweep_int, '--', f'crop interior ({INTERIOR_FRAC:g} r)')):
        if not sweep:
            continue
        xs = sorted(sweep)
        for key, c, lbl in (('seedseed_hp', FIG_COLORS['seedseed'], 'model vs model'),
                            ('seedexp_hp', FIG_COLORS['seedexp'], 'model vs experiment'),
                            ('null_hp', FIG_COLORS['null'], 'null')):
            ax.plot(xs, [sweep[s][key].mean() for s in xs], style, marker='o',
                    ms=3.5, color=c, label=f'{lbl}, {tag}')
    ax.set_xlabel('Gaussian sigma (px)')
    ax.set_ylabel('mean correlation')
    ax.set_title('Robustness of the split\nand crop-boundary control')
    ax.axhline(0, color='0.7', lw=0.8, zorder=0)
    ax.legend(fontsize=6.5, frameon=False)

    fig.suptitle(f'Reproducibility of {model_type} input gradients across training '
                 f'runs ({ds.label}, one point per neuron)')
    fig.tight_layout()
    plots.save_figure(fig, cfg.figure_dir('extra') /
                      f'gradient_reproducibility_{model_type}_{ds.key}.png', dpi=150)


# -----------------------------------------------------------------------------
# Drivers
# -----------------------------------------------------------------------------

def cache_file(ds, model_type):
    return cfg.cache_dir(ds) / f'gradient_reproducibility_{model_type}.npz'


def run_dataset(ds, model_type, sigmas, headline, device, interior=True):
    banner(f'gradient reproducibility | {ds.label} | {model_type}')
    grads, names = L.run_gradients(model_type, ds, device)
    print(f'  gradients: {grads.shape}  (runs, neurons, images, px, px)')

    exp_frame = experimental_frame(ds)
    boxes = L.rf_bboxes(dat.lsta_reference(ds)['ellipses'], ds=ds)
    masks = interior_masks(ds, model_type=model_type) if interior else None
    if masks is not None:
        print(f'  interior control keeps {masks.sum(axis=(1, 2)).mean():.0f} px/neuron')

    sweep_full, sweep_int, store = {}, {}, {}
    for sigma in sigmas:
        res_f = analyse(grads, exp_frame, boxes, sigma)
        sweep_full[sigma] = per_neuron(res_f)
        for k, v in res_f.items():
            store[f'full_s{sigma:g}_{k}'] = v
        if masks is not None:
            res_i = analyse(grads, exp_frame, boxes, sigma, masks=masks)
            sweep_int[sigma] = per_neuron(res_i)
            for k, v in res_i.items():
                store[f'interior_s{sigma:g}_{k}'] = v
        print(f'  sigma {sigma:>4.1f} done', flush=True)

    head = min(sweep_full, key=lambda s: abs(s - headline))
    print_block(f'{ds.label} -- full RF box, sigma = {head:g}', sweep_full[head])
    if sweep_int:
        print_block(f'{ds.label} -- crop interior only, sigma = {head:g}', sweep_int[head])
    print_sweep(sweep_full, f'{ds.label}, full RF box')

    out = cfg.ensure(cfg.cache_dir(ds)) / cache_file(ds, model_type).name
    np.savez_compressed(out, sigmas=np.array(sigmas), run_names=np.array(names),
                        interior_frac=INTERIOR_FRAC, **store)
    print(f'\n  cache -> {out}')
    make_figure(sweep_full[head], sweep_full, sweep_int, head, len(names), ds, model_type)
    return head, sweep_full, sweep_int


def load_cached(ds, model_type, headline):
    path = cache_file(ds, model_type)
    if not path.exists():
        raise SystemExit(f'no cache at {path}; run without --from-cache first')
    d = np.load(path)
    sweeps = {'full': {}, 'interior': {}}
    for key in d.files:
        if not key.startswith(('full_s', 'interior_s')):
            continue
        tag, rest = key.split('_s', 1)
        sigma_str, metric = rest.split('_', 1)
        sweeps[tag].setdefault(float(sigma_str), {})[metric] = d[key].mean(axis=0)
    head = min(sweeps['full'], key=lambda s: abs(s - headline))
    n_runs = len(d['run_names']) if 'run_names' in d.files else 0
    return head, sweeps['full'], sweeps['interior'], n_runs


def fine_structure_table(ds, device, sigmas=TABLE_SIGMAS,
                         models=('transformer', 'cnn', 'GP')):
    """Model-vs-model agreement of the fine structure, per model and sigma."""
    banner(f'fine-structure reproducibility | {ds.label}')
    exp_boxes = L.rf_bboxes(dat.lsta_reference(ds)['ellipses'], ds=ds)
    results = {}
    for m in models:
        try:
            grads, _ = L.run_gradients(m, ds, device, verbose=False)
        except SystemExit as exc:
            print(f'  [skip] {m}: {exc}')
            continue
        S, N, I = grads.shape[:3]
        pairs = list(itertools.combinations(range(S), 2))
        rng = np.random.default_rng(0)
        partner = np.array([rng.choice([k for k in range(N) if k != n]) for n in range(N)])
        per_sigma = {}
        for sigma in sigmas:
            maps = L.highpass(grads, sigma) if sigma > 0 else grads
            ss, nu = [], []
            for n in range(N):
                y0, y1, x0, x1 = exp_boxes[n]
                if y1 <= y0 or x1 <= x0:
                    continue
                a = maps[:, n, :, y0:y1, x0:x1]
                b = maps[:, partner[n], :, y0:y1, x0:x1]
                ss.append(np.mean([[L.correlation(a[u, i], a[v, i]) for i in range(I)]
                                   for u, v in pairs]))
                nu.append(np.mean([[L.correlation(a[u, i], b[v, i]) for i in range(I)]
                                   for u, v in pairs]))
            per_sigma[sigma] = (float(np.mean(ss)), float(np.mean(nu)))
        results[m] = per_sigma
        print(f'  {m} done')

    if not results:
        return
    present = list(results)
    print(f'\n{"sigma":>8}' + ''.join(f'{cfg.MODEL_ABBR[m]:>9}' for m in present)
          + f'{"floor":>9}')
    for sigma in sigmas:
        floor = np.mean([results[m][sigma][1] for m in present])
        print(f'{("none" if sigma <= 0 else f"{sigma:g}"):>8}'
              + ''.join(f'{results[m][sigma][0]:>9.3f}' for m in present)
              + f'{floor:>9.3f}')
    out = cfg.ensure(cfg.cache_dir(ds)) / 'fine_structure_table.npz'
    np.savez_compressed(out, sigmas=np.array(sigmas), models=np.array(present),
                        values=np.array([[results[m][s][0] for s in sigmas] for m in present]),
                        floor=np.array([[results[m][s][1] for s in sigmas] for m in present]))
    print(f'  -> {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(ap)
    add_device_arg(ap, default='cpu')
    ap.add_argument('--model', default='transformer',
                    choices=('transformer', 'cnn', 'GP'))
    ap.add_argument('--sigma', type=float, default=None,
                    help='single sigma instead of the default sweep')
    ap.add_argument('--no-interior', action='store_true',
                    help='skip the crop-boundary control (halves the runtime)')
    ap.add_argument('--from-cache', action='store_true',
                    help='rebuild the figures without recomputing gradients')
    ap.add_argument('--tables', action='store_true',
                    help='fine-structure agreement for all three models')
    args = ap.parse_args()

    device = device_from(args, default='cpu')
    sigmas = (args.sigma,) if args.sigma is not None else SIGMA_SWEEP
    headline = args.sigma if args.sigma is not None else 2.0

    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        if not require_lsta(ds):
            continue
        if args.tables:
            fine_structure_table(ds, device)
        elif args.from_cache:
            head, full, interior, n_runs = load_cached(ds, args.model, headline)
            print_block(f'{ds.label} -- full RF box, sigma = {head:g}', full[head])
            make_figure(full[head], full, interior, head, n_runs, ds, args.model)
        else:
            run_dataset(ds, args.model, sigmas, headline, device,
                        interior=not args.no_interior)


if __name__ == '__main__':
    main()
