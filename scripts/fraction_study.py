#!/usr/bin/env python
"""Data-efficiency benchmark: train every model on 10%, 20% ... 100% of the data.

    python scripts/fraction_study.py --model transformer
    python scripts/fraction_study.py --model all --dataset salamander
    python scripts/fraction_study.py --fractions 0.1 0.5 1.0 --folds 0 1
    python scripts/fraction_study.py --seeds 0 1 2          # 3 seeds per fold

Each (model, fraction, fold, seed) is one training run, written to

    results/<dataset>/fraction_study/<model>/frac_XXX/fold_K/seed_S/

The sweep is the same K-fold cross-validation as scripts/train.py, at every
fraction: fold K is held out (early stopping, out-of-fold score) and the
training folds are subsampled to the fraction, reproducibly from `seed`. The
held-out fold does not depend on the fraction, so a low-data run and a
full-data run of the same fold are stopped on the same stimuli. `seed` also
seeds initialisation and batch order; at fraction 1.0 runs of the same fold
differ only in that.

By default one seed per fold, so a point is N_FOLDS runs. The run at 100 % with
the canonical seed (cirf.config.TRAIN_SEED) is the same training as the fold's
canonical checkpoint under results/<dataset>/checkpoints/<model>/fold_K/, so it
is written to both places, by whichever of this script and scripts/train.py
runs first; the other finds it and skips it.

This is the expensive script in the repository: the default sweep is
10 fractions x 5 folds x 3 models. Use --fractions / --folds / --seeds / --model
to cut it down, and note that it resumes -- runs whose checkpoint already
exists are skipped unless --force is given.

Score the sweep with scripts/make_paper_figures.py or `cirf.evaluate.fraction_metrics`.
"""

from __future__ import annotations

import argparse
import time

from _common import (add_dataset_arg, add_device_arg, banner, cfg,
                     datasets_from, device_from)
from cirf import checkpoints as ckpt

MODELS = ('transformer', 'cnn', 'GP')


def run_one(model_type, ds, frac, fold, seed, out_dir, device, args):
    common = dict(out_dir=out_dir, fold=fold, n_folds=args.n_folds,
                  data_fraction=frac, seed=seed, device=device, verbose=False)
    if model_type == 'transformer':
        from models.transformer import get_cfg, train
        return train(ds, cfg=get_cfg(ds), max_epochs=args.epochs, **common)
    if model_type == 'cnn':
        from models.cnn import get_cfg, train
        return train(ds, cfg=get_cfg(ds), max_epochs=args.epochs, **common)
    if model_type == 'GP':
        from models.gp.train import get_cfg, train
        return train(ds, cfg=get_cfg(ds), **common)
    raise ValueError(model_type)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='all', choices=(*MODELS, 'all'))
    add_dataset_arg(ap)
    add_device_arg(ap)
    ap.add_argument('--fractions', type=float, nargs='+', default=cfg.FRACTIONS,
                    help=f'default: {cfg.FRACTIONS}')
    ap.add_argument('--folds', type=int, nargs='+', default=None,
                    help='folds to run (default: all of them)')
    ap.add_argument('--n-folds', type=int, default=cfg.N_FOLDS,
                    help=f'K of the K-fold cross-validation (default: {cfg.N_FOLDS})')
    ap.add_argument('--seeds', type=int, nargs='+', default=cfg.FRAC_SEEDS,
                    help=f'training seeds per fold (default: {cfg.FRAC_SEEDS})')
    ap.add_argument('--epochs', type=int, default=None,
                    help='cap on the number of epochs per run')
    ap.add_argument('--force', action='store_true',
                    help='retrain runs whose checkpoint already exists')
    args = ap.parse_args()

    device = device_from(args)
    models = MODELS if args.model == 'all' else (args.model,)
    folds = list(range(args.n_folds)) if args.folds is None else args.folds

    def is_canonical(frac, seed):
        return (frac == 1.0 and seed == cfg.TRAIN_SEED
                and args.n_folds == cfg.N_FOLDS)

    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        for model_type in models:
            banner(f'fraction study | {model_type} | {ds.label} | '
                   f'{len(folds)} folds x {len(args.seeds)} seeds')
            for frac in args.fractions:
                for fold in folds:
                    for seed in args.seeds:
                        out_dir = cfg.fraction_dir(model_type, frac, fold, seed, ds)
                        twin = cfg.fold_dir(model_type, fold, ds) \
                            if is_canonical(frac, seed) else None
                        if not args.force:
                            if ckpt.exists(model_type, out_dir):
                                print(f'  [keep] frac {frac:>4.0%} fold {fold} seed {seed}')
                                continue
                            if twin is not None and ckpt.exists(model_type, twin):
                                ckpt.mirror(twin, out_dir)
                                print(f'  [keep] frac {frac:>4.0%} fold {fold} seed {seed}'
                                      f'  (the canonical fold, copied from {twin})')
                                continue
                        t0 = time.time()
                        res = run_one(model_type, ds, frac, fold, seed, out_dir,
                                      device, args)
                        val = res.get('best_val')
                        print(f'  frac {frac:>4.0%} fold {fold} seed {seed}: '
                              + (f'val {val:.4f}  ' if val is not None else '')
                              + f'({time.time() - t0:.0f}s)  -> {out_dir}',
                              flush=True)
                        if twin is not None:
                            ckpt.mirror(out_dir, twin)
                            print(f'  also the canonical fold {fold} -> {twin}')


if __name__ == '__main__':
    main()
