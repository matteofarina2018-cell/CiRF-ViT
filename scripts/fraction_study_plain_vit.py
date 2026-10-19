#!/usr/bin/env python
"""Data efficiency of the plain ViT (the `no_crop` variant).

    python scripts/fraction_study_plain_vit.py --dataset mice --fraction 0.3 --fold 2

One run = one (fraction, fold, seed): the K-fold partition of the repository,
the training folds subsampled to the fraction exactly as scripts/fraction_study.py
does (same `fold_indices`, same seed), so the curve sits next to the CiRF-ViT
and CNN curves point by point. Weights go to
results/<dataset>/fraction_study/plain_vit/frac_XXX/fold_K/seed_S/, and the
test metrics of the run to metrics.json in the same directory. A run whose
metrics.json exists is skipped.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from _common import add_dataset_arg, add_device_arg, cfg, datasets_from, device_from
from cirf import data as dat
from cirf.metrics import (adjusted_r2_per_neuron, cc_norm_per_neuron,
                          explained_variance_per_neuron)
from models.transformer import fit, get_cfg, save_checkpoint
from models.transformer.variants import build_variant


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_dataset_arg(ap, allow_both=False)
    add_device_arg(ap)
    ap.add_argument('--fraction', type=float, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=cfg.TRAIN_SEED)
    ap.add_argument('--n-folds', type=int, default=cfg.N_FOLDS)
    ap.add_argument('--variant', default='no_crop')
    args = ap.parse_args()

    device = device_from(args)
    ds = datasets_from(args)[0]
    cfg.set_dataset(ds)
    name = 'plain_vit' if args.variant == 'no_crop' else args.variant
    out = cfg.ensure(cfg.fraction_root(name, ds) / f'frac_{round(args.fraction * 100):03d}'
                     / f'fold_{args.fold}' / f'seed_{args.seed}')
    if (out / 'metrics.json').exists():
        print(f'[keep] {out}')
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    conf = {**get_cfg(ds), 'RUN_FOLD': args.fold, 'RUN_SEED': args.seed,
            'RUN_N_FOLDS': args.n_folds}
    tr, vl, _ = dat.get_dataloaders(ds, batch_size=conf['BATCH_SIZE'],
                                    data_fraction=args.fraction,
                                    random_seed=args.seed, fold=args.fold,
                                    n_folds=args.n_folds)
    t0 = time.time()
    model = build_variant(args.variant, conf, ds, device)
    n_par = sum(p.numel() for p in model.parameters())
    best_val, hist, n_ep = fit(model, tr, vl, conf, device, verbose=False)
    save_checkpoint(model, conf, out, hist, dataset=ds.key, variant=args.variant,
                    seed=args.seed, fold=args.fold, n_folds=args.n_folds,
                    data_fraction=args.fraction, val_loss=best_val, n_params=n_par)

    from cirf.evaluate import predict_torch
    y = predict_torch(model, dat.images('test', ds), device)
    rep = dat.test_repeats(ds)
    m = {'dataset': ds.key, 'variant': args.variant, 'fraction': args.fraction,
         'fold': args.fold, 'seed': args.seed, 'n_train': len(tr.dataset),
         'epochs': n_ep, 'val_loss': best_val, 'n_params': n_par,
         'seconds': round(time.time() - t0, 1),
         'ev': explained_variance_per_neuron(y, rep).tolist(),
         'adj_r2': adjusted_r2_per_neuron(y, rep).tolist(),
         'cc_norm': cc_norm_per_neuron(y, rep).tolist()}
    (out / 'metrics.json').write_text(json.dumps(m), encoding='utf-8')
    print(f'{ds.key} {name} frac {args.fraction:.0%} fold {args.fold}: '
          f'EV {np.mean(m["ev"]):.4f} CCnorm {np.mean(m["cc_norm"]):.4f} '
          f'ep {n_ep} ({m["seconds"]:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
