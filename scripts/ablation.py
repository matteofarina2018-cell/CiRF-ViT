#!/usr/bin/env python
"""Ablate the architecture one component at a time, several runs each.

    python scripts/ablation.py                          # all variants, mouse
    python scripts/ablation.py --dataset both --folds 0 1 2 3 4
    python scripts/ablation.py --variants no_crop focal --seeds 0 1 2

Variants (see models/transformer/variants.py):

    no_crop      every neuron attends to the whole feature map
    no_block     no transformer at all: tokeniser + crop + readout
    no_rope      attention without positional information
    no_attn      blocks without token mixing (MLP only)
    focal        the crop replaced by ViT-P's learnable focal attention bias,
                 i.e. locality learned in the attention instead of measured
    readout_crop global attention, the measured circle only in the readout's
                 softmax: the spatial prior without the constraint on attention
    cnn_crop     the CNN baseline with its readout confined to the same circle:
                 the constraint without the transformer
    full         the model itself, if you want it retrained alongside

Further variants (convolutional core, crop centres, radius, sampling) are in
models/transformer/extra_variants.py and are run the same way, by name.

The last two separate the crop's two roles. `no_crop` removes both the
restriction on what a neuron attends to and the readout's knowledge of where
the neuron is, so its drop alone does not say which of the two matters;
`readout_crop` keeps the second and drops the first, `cnn_crop` keeps the
constraint and swaps the backbone. Compare `readout_crop` with the full model
and `no_crop`, and `cnn_crop` with the CNN baseline (scripts/train.py).

Every variant is trained by the same K-fold cross-validation as the model
itself: one run per (fold, seed), fold K held out, the default being one seed
per fold. Per-neuron metrics are appended to
results/<dataset>/experiments/ablation.json after every run, so an interrupted
sweep keeps what it had -- the figures read that file -- and every run's
weights go to results/<dataset>/experiments/ablation/<variant>/fold_K/seed_S/,
in the layout of the canonical checkpoints, so a variant can be rebuilt with
`cirf.checkpoints.load_torch_model('transformer', that_dir)`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from _common import (add_dataset_arg, add_device_arg, banner, cfg,
                     datasets_from, device_from)
from cirf import data as dat
from cirf.metrics import (adjusted_r2_per_neuron, cc_norm_per_neuron,
                          explained_variance_per_neuron)
from models.transformer import fit, get_cfg, save_checkpoint
from models.transformer.variants import VARIANT_LABELS, build_variant, cnn_crop_cfg

DEFAULT_VARIANTS = ('no_crop', 'readout_crop', 'no_block', 'no_rope', 'no_attn',
                    'focal', 'cnn_crop')


def evaluate(model, ds, device):
    """(EV, adj R^2) per neuron on the test set."""
    from cirf.evaluate import predict_torch
    y_pred = predict_torch(model, dat.images('test', ds), device)
    rep = dat.test_repeats(ds)
    return (explained_variance_per_neuron(y_pred, rep),
            adjusted_r2_per_neuron(y_pred, rep), cc_norm_per_neuron(y_pred, rep))


def run_transformer_variant(name, conf, ds, device, fold, n_folds, seed, epochs):
    """Train one CiRF-ViT variant; returns (model, best val, epochs run, n_params)."""
    tr, vl, _ = dat.get_dataloaders(ds, batch_size=conf['BATCH_SIZE'],
                                    fold=fold, n_folds=n_folds)
    conf = {**conf, 'RUN_FOLD': fold, 'RUN_SEED': seed, 'RUN_N_FOLDS': n_folds}
    model = build_variant(name, conf, ds, device)
    n_par = sum(p.numel() for p in model.parameters())
    best_val, hist, n_ep = fit(model, tr, vl, conf, device, epochs=epochs,
                               verbose=False)
    save_checkpoint(model, conf, cfg.ablation_dir(name, fold, seed, ds), hist,
                    dataset=ds.key, variant=name, seed=seed, fold=fold,
                    n_folds=n_folds, val_loss=best_val, n_params=n_par)
    return model, best_val, n_ep, n_par


def run_cnn_crop(conf, ds, device, fold, n_folds, seed, epochs):
    """Train the CNN with the crop on its readout, by the CNN's own trainer.

    Everything about the run -- objective, schedule, early stopping, the
    checkpoint layout -- is the CNN baseline's, so the only difference from
    the canonical CNN fold is the circle; `variant` is recorded in
    `config.json` so the checkpoint loader rebuilds it as such.
    """
    from models.cnn import train as cnn_train

    out_dir = cfg.ablation_dir('cnn_crop', fold, seed, ds)
    res = cnn_train(ds, cfg=cnn_crop_cfg(conf, ds), out_dir=out_dir, seed=seed,
                    device=device, max_epochs=epochs, fold=fold, n_folds=n_folds,
                    verbose=False)
    info_path = out_dir / 'config.json'
    info = json.loads(info_path.read_text(encoding='utf-8'))
    info['variant'] = 'cnn_crop'
    info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
    return (res['model'], res['best_val'], len(res['history']['val_loss']),
            res['n_params'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variants', nargs='+', default=list(DEFAULT_VARIANTS))
    ap.add_argument('--folds', type=int, nargs='+', default=None,
                    help='folds to run (default: all of them)')
    ap.add_argument('--n-folds', type=int, default=cfg.N_FOLDS,
                    help=f'K of the K-fold cross-validation (default: {cfg.N_FOLDS})')
    ap.add_argument('--seeds', type=int, nargs='+', default=cfg.FRAC_SEEDS,
                    help=f'training seeds per fold (default: {cfg.FRAC_SEEDS})')
    add_dataset_arg(ap)
    add_device_arg(ap)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--out', default=None, help='JSON file (default: '
                                                'results/<dataset>/experiments/ablation.json)')
    args = ap.parse_args()

    device = device_from(args)
    folds = list(range(args.n_folds)) if args.folds is None else args.folds

    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        conf = get_cfg(ds)
        out_path = Path(args.out) if args.out else \
            cfg.ensure(cfg.experiment_dir(ds)) / 'ablation.json'
        store = {}
        if out_path.exists():
            store = json.loads(out_path.read_text(encoding='utf-8'))

        banner(f'ablation | {ds.label} | {dat.n_cells(ds)} neurons | '
               f'{len(folds)} folds x {len(args.seeds)} seeds')
        for name in args.variants:
            entry = store.setdefault(name, {'ev': [], 'adj_r2': [], 'seeds': []})
            if 'folds' not in entry:
                if entry['seeds']:
                    raise SystemExit(
                        f'{out_path} holds {name!r} runs without a fold: they '
                        f'predate the K-fold cross-validation. Delete the file '
                        f'(or that variant) and rerun.')
                entry['folds'] = []
            done = set(zip(entry['folds'], entry['seeds']))
            for fold in folds:
                for seed in args.seeds:
                    if (fold, seed) in done:
                        print(f'  [keep] {name:9s} fold {fold} seed {seed}')
                        continue
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    t0 = time.time()
                    runner = run_cnn_crop if name == 'cnn_crop' else \
                        lambda *a: run_transformer_variant(name, *a)
                    model, best_val, epochs, n_par = runner(
                        conf, ds, device, fold, args.n_folds, seed, args.epochs)
                    ev, adj, ccn = evaluate(model, ds, device)
                    entry.setdefault('cc_norm', []).append(ccn.tolist())
                    entry.setdefault('n_params', []).append(int(n_par))
                    entry.setdefault('epochs', []).append(int(epochs))
                    entry.setdefault('seconds', []).append(round(time.time() - t0, 1))
                    entry['ev'].append(ev.tolist())
                    entry['adj_r2'].append(adj.tolist())
                    entry['folds'].append(fold)
                    entry['seeds'].append(seed)
                    print(f'  {name:9s} fold {fold} seed {seed}  ep{epochs:3d}  '
                          f'val {best_val:.4f}  EV {ev.mean():.4f}  '
                          f'adjR2 {adj.mean():.4f}  CCnorm {ccn.mean():.4f}  '
                          f'({n_par:,} par, {time.time() - t0:.0f}s)', flush=True)
                    out_path.write_text(json.dumps(store), encoding='utf-8')

        print(f'\n  {"variant":>12}  {"EV mean":>9}  {"EV sd":>7}')
        for name in args.variants:
            if not store.get(name, {}).get('ev'):
                continue
            per_run = np.array(store[name]['ev']).mean(axis=1)
            print(f'  {VARIANT_LABELS.get(name, name).splitlines()[0]:>12}  '
                  f'{per_run.mean():9.4f}  {per_run.std():7.4f}')
        print(f'  -> {out_path}')


if __name__ == '__main__':
    main()
