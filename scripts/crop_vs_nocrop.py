#!/usr/bin/env python
"""Training dynamics with and without the per-neuron crop.

    python scripts/crop_vs_nocrop.py --dataset both

Trains the no-crop variant on one fold of the cross-validation (--fold,
default 0), for exactly as many epochs as the canonical model took on that
fold, with early stopping switched off so both curves span the same range, and
stores its history (and the fold it belongs to) next to the canonical one,
and its weights (those of its best validation epoch, like every checkpoint):

    results/<dataset>/experiments/history_nocrop.json
    results/<dataset>/experiments/crop_vs_nocrop/fold_K/

What the figure then shows is the point of the experiment: with the crop the
validation loss settles close to the training loss, without it the training
loss keeps falling while the validation loss stalls -- the overfitting that
appears once the locality prior is removed and attention has to discover where
to look from limited data.

Needs that fold of the canonical CiRF-ViT run (scripts/train.py --model
transformer), whose history.json fixes the epoch budget.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from _common import (add_dataset_arg, add_device_arg, banner, cfg,
                     datasets_from, device_from)
from cirf import data as dat
from models.transformer import fit, get_cfg, save_checkpoint
from models.transformer.variants import build_nocrop_model

SEED = 42


def early_stop_epoch(val_loss, patience):
    """Epoch (0-based) early stopping would have picked -- not the global min."""
    best, best_ep, waited = float('inf'), 0, 0
    for i, v in enumerate(val_loss):
        if v < best:
            best, best_ep, waited = v, i, 0
        else:
            waited += 1
            if waited >= patience:
                break
    return best_ep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(ap)
    add_device_arg(ap)
    ap.add_argument('--epochs', type=int, default=None,
                    help='epoch budget (default: as long as the crop model ran)')
    ap.add_argument('--fold', type=int, default=0,
                    help='fold of the cross-validation to train on (default: 0)')
    ap.add_argument('--n-folds', type=int, default=cfg.N_FOLDS,
                    help=f'K of the K-fold cross-validation (default: {cfg.N_FOLDS})')
    args = ap.parse_args()

    device = device_from(args)

    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        conf = get_cfg(ds)
        crop_hist_path = cfg.fold_dir('transformer', args.fold, ds) / 'history.json'
        if not crop_hist_path.exists():
            print(f'  [skip] {ds.label}: train fold {args.fold} of the canonical '
                  f'model first ({crop_hist_path} is missing)')
            continue
        crop_hist = json.loads(crop_hist_path.read_text(encoding='utf-8'))
        epochs = int(args.epochs or len(crop_hist['val_loss']))

        banner(f'crop vs no-crop | {ds.label} | fold {args.fold} | {epochs} epochs')
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        tr, vl, _ = dat.get_dataloaders(ds, batch_size=conf['BATCH_SIZE'],
                                        fold=args.fold, n_folds=args.n_folds)
        model = build_nocrop_model(conf, ds, device)
        print(f'  no-crop model: {sum(p.numel() for p in model.parameters()):,} '
              f'parameters (crop model attends to a circle, this one to all '
              f'{model.fh * model.fh} tokens)')
        best, hist, _ = fit(model, tr, vl, conf, device, early_stop=False,
                            epochs=epochs, verbose=True)

        hist['fold'] = args.fold
        hist['n_folds'] = args.n_folds
        out = cfg.ensure(cfg.experiment_dir(ds)) / 'history_nocrop.json'
        out.write_text(json.dumps(hist, indent=2), encoding='utf-8')
        ckpt_dir = save_checkpoint(
            model, conf, cfg.nocrop_dir(args.fold, ds), hist, dataset=ds.key,
            variant='no_crop', seed=SEED, fold=args.fold, n_folds=args.n_folds,
            epochs=epochs, val_loss=best,
            n_params=sum(p.numel() for p in model.parameters()))

        for tag, h in (('crop', crop_hist), ('no-crop', hist)):
            i = early_stop_epoch(h['val_loss'], conf['EARLY_STOP'])
            print(f'  {tag:>8}: min train {min(h["train_loss"]):.4f}  '
                  f'early-stop pick epoch {i + 1} val {h["val_loss"][i]:.4f}  '
                  f'final gap {h["val_loss"][-1] - h["train_loss"][-1]:+.4f}')
        print(f'  -> {out}\n  weights -> {ckpt_dir}')


if __name__ == '__main__':
    main()
