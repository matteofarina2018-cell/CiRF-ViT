#!/usr/bin/env python
"""Train one model (or all four) on a dataset, by K-fold cross-validation.

    python scripts/train.py --model transformer                  # CiRF-ViT, mouse
    python scripts/train.py --model all --dataset salamander
    python scripts/train.py --model transformer --data my.npz    # your recording
    python scripts/train.py --model cnn --folds 0 1              # two folds only
    python scripts/train.py --model cnn --seed 3 --fraction 0.2  # a low-data run

The stimulus pool is partitioned once, with a fixed seed, into K folds
(cirf.config.N_FOLDS, --n-folds); every model is fitted K times, each time on
K-1 folds with the remaining fold held out for early stopping and the
out-of-fold score. Fold k goes to results/<dataset>/checkpoints/<model>/fold_k/
unless --out says otherwise (fold_k/ is created inside it). Score the folds
afterwards with scripts/evaluate.py, which also aggregates them.

A fold whose checkpoint is already there is kept, not retrained (--force to
retrain). A run at 100 % of the data with the canonical seed is the same
training as the data-efficiency sweep's run at that point, so it is written to
both places: here and under results/<dataset>/fraction_study/<model>/frac_100/.
scripts/fraction_study.py does the same the other way round, and skips it.

The models are independent: the GP and the LN fit each neuron separately, the
CNN and CiRF-ViT share a core across the population.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _common import add_dataset_arg, add_device_arg, banner, cfg, datasets_from, device_from
from cirf import checkpoints as ckpt

MODELS = ('transformer', 'cnn', 'GP', 'LN')
SWEEP_MODELS = ('transformer', 'cnn', 'GP')     # the ones fraction_study.py runs


def train_one(model_type, ds, args, device, fold, out_dir):
    common = dict(out_dir=out_dir, fold=fold, n_folds=args.n_folds,
                  verbose=not args.quiet)
    if model_type == 'transformer':
        from models.transformer import get_cfg, train
        conf = get_cfg(ds, **{'DEFAULT_RADIUS': args.radius})
        if args.radius is not None:
            conf['NEURON_RADII'] = None      # a uniform radius was requested
        return train(ds, cfg=conf, seed=args.seed, data_fraction=args.fraction,
                     device=device, max_epochs=args.epochs, **common)
    if model_type == 'cnn':
        from models.cnn import get_cfg, train
        return train(ds, cfg=get_cfg(ds), seed=args.seed,
                     data_fraction=args.fraction, device=device,
                     max_epochs=args.epochs, **common)
    if model_type == 'LN':
        from models.ln import get_cfg, train
        return train(ds, cfg=get_cfg(ds), device=args.device or 'cpu',
                     cells=args.cells, n_repeats=args.repeats, seed=args.seed,
                     data_fraction=args.fraction, **common)
    if model_type == 'GP':
        from models.gp.train import get_cfg, train
        return train(ds, cfg=get_cfg(ds), cells=args.cells, device=device,
                     data_fraction=args.fraction, seed=args.seed, **common)
    raise ValueError(model_type)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='transformer',
                    choices=(*MODELS, 'all'), help='which model to train')
    add_dataset_arg(ap)
    add_device_arg(ap)
    ap.add_argument('--out', default=None,
                    help='output root; fold_K/ is created inside it '
                         '(default: results/<dataset>/checkpoints/<model>)')
    ap.add_argument('--folds', type=int, nargs='+', default=None,
                    help='which folds to train (default: all of them)')
    ap.add_argument('--n-folds', type=int, default=cfg.N_FOLDS,
                    help=f'K of the K-fold cross-validation (default: {cfg.N_FOLDS})')
    ap.add_argument('--seed', type=int, default=cfg.TRAIN_SEED,
                    help='random seed; also picks the subset when --fraction < 1 '
                         f'(default: {cfg.TRAIN_SEED}, the canonical run)')
    ap.add_argument('--fraction', type=float, default=1.0,
                    help='share of the training folds to use (default: 1.0)')
    ap.add_argument('--epochs', type=int, default=None,
                    help='cap on the number of epochs (default: the model config)')
    ap.add_argument('--radius', type=int, default=None,
                    help='CiRF-ViT: uniform crop radius in px, instead of the '
                         'tuned per-neuron radii')
    ap.add_argument('--cells', type=int, nargs='+', default=None,
                    help='GP/LN: fit only these neurons')
    ap.add_argument('--repeats', type=int, default=None,
                    help='LN: fits per neuron (default: config N_REPEATS)')
    ap.add_argument('--force', action='store_true',
                    help='retrain folds whose checkpoint already exists')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    device = device_from(args)
    models = MODELS if args.model == 'all' else (args.model,)
    folds = list(range(args.n_folds)) if args.folds is None else args.folds
    bad = [k for k in folds if not 0 <= k < args.n_folds]
    if bad:
        ap.error(f'--folds {bad} outside [0, {args.n_folds})')

    # A canonical run (default location, all the data, canonical seed, every
    # neuron) is also the sweep's run at 100 %: write it there too.
    canonical = (args.out is None and args.fraction == 1.0
                 and args.seed == cfg.TRAIN_SEED and args.cells is None
                 and args.n_folds == cfg.N_FOLDS)

    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        for model_type in models:
            root = Path(args.out) if args.out else cfg.checkpoint_dir(model_type, ds)
            summary = []
            for fold in folds:
                out_dir = root / f'fold_{fold}'
                if ckpt.exists(model_type, out_dir) and not args.force:
                    print(f'  [keep] {model_type} fold {fold}: {out_dir} '
                          f'(--force to retrain)')
                    continue
                banner(f'{model_type} | {ds.label} | fold {fold}/{args.n_folds}'
                       f' | fraction {args.fraction:.0%}')
                t0 = time.time()
                res = train_one(model_type, ds, args, device, fold, out_dir)
                val = res.get('best_val')
                summary.append((fold, val, time.time() - t0))
                print(f'  {model_type} fold {fold} done in {time.time() - t0:.0f}s')
                if canonical and model_type in SWEEP_MODELS:
                    twin = ckpt.mirror(out_dir, cfg.fraction_dir(
                        model_type, 1.0, fold, args.seed, ds))
                    print(f'  also the data-efficiency run at 100% -> {twin}')
            print(f'\n  {model_type} | {ds.label} | {len(folds)} folds -> {root}')
            for fold, val, secs in summary:
                print(f'    fold {fold}: '
                      + (f'held-out loss {val:.4f}  ' if val is not None else '')
                      + f'({secs:.0f}s)')


if __name__ == '__main__':
    main()
