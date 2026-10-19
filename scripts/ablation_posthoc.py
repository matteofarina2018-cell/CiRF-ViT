#!/usr/bin/env python
"""Post-hoc analyses of the ablation checkpoints.

    python scripts/ablation_posthoc.py --dataset mice --variant centers_random

Waits until every fold of the variant has a checkpoint (the ablation may still
be running), then, per fold:

  * the full evaluation of the repository (`cirf.evaluate.evaluate_fold`):
    eval_data.npz next to the checkpoint, with the LSTA agreement (does the
    LSTA match survive without measured centres?);
  * for every model with a crop, where the centres went: initial
    centre (what the variant started from), trained centre, measured STA
    centre, in stimulus pixels and in units of the neuron's radius.

Summary in results/<dataset>/experiments/posthoc_<variant>.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from _common import add_dataset_arg, add_device_arg, cfg, datasets_from, device_from
from cirf import checkpoints as ckpt
from cirf import data as dat
from cirf.metrics import cc_norm_per_neuron


def centre_stats(model_dir, variant, ds, device):
    """Per-neuron initial / trained / measured centres, or None without a crop."""
    from models.transformer.variants import build_variant

    info = ckpt.load_run_info(model_dir)
    conf = dict(info['cfg'])
    model, _ = ckpt.load_torch_model('transformer', model_dir, ds, device)
    circle = getattr(model, 'circle', None)
    if circle is None:
        return None
    stride = conf.get('CNN_STRIDE', 1)
    init = build_variant(variant, conf, ds, 'cpu').circle
    cx0, cy0 = init.cx.detach().numpy() * stride, init.cy.detach().numpy() * stride
    cx1 = circle.cx.detach().cpu().numpy() * stride
    cy1 = circle.cy.detach().cpu().numpy() * stride
    mx, my = (np.asarray(a, np.float64) for a in dat.centers(ds))
    radius = np.asarray(conf['NEURON_RADII'], np.float64)
    if variant.startswith('radius_x'):
        from models.transformer.extra_variants import RADIUS_SCALES
        radius = np.maximum(2, np.round(radius * RADIUS_SCALES[variant]))
    moved = np.hypot(cx1 - cx0, cy1 - cy0)
    err0 = np.hypot(cx0 - mx, cy0 - my)
    err1 = np.hypot(cx1 - mx, cy1 - my)
    return {'init_x': cx0.tolist(), 'init_y': cy0.tolist(),
            'trained_x': cx1.tolist(), 'trained_y': cy1.tolist(),
            'radius': radius.tolist(),
            'moved_px': moved.tolist(), 'moved_rad': (moved / radius).tolist(),
            'dist_to_sta_init_px': err0.tolist(), 'dist_to_sta_trained_px': err1.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_dataset_arg(ap, allow_both=False)
    add_device_arg(ap)
    ap.add_argument('--variant', required=True)
    ap.add_argument('--n-folds', type=int, default=cfg.N_FOLDS)
    ap.add_argument('--seed', type=int, default=cfg.TRAIN_SEED)
    ap.add_argument('--wait-hours', type=float, default=72.0)
    args = ap.parse_args()

    device = device_from(args)
    ds = datasets_from(args)[0]
    cfg.set_dataset(ds)
    model_type = 'cnn' if args.variant in ('cnn', 'cnn_crop') else 'transformer'
    if args.variant == 'cnn':
        dirs = [cfg.fold_dir('cnn', k, ds) for k in range(args.n_folds)]
    else:
        dirs = [cfg.ablation_dir(args.variant, k, args.seed, ds)
                for k in range(args.n_folds)]

    t_end = time.time() + 3600 * args.wait_hours
    while not all((d / 'model.pt').exists() and (d / 'config.json').exists()
                  for d in dirs):
        if time.time() > t_end:
            raise SystemExit(f'{args.variant} {ds.key}: folds still missing, giving up')
        time.sleep(120)
    time.sleep(30)          # let the last checkpoint finish writing

    out = cfg.ensure(cfg.experiment_dir(ds)) / f'posthoc_{args.variant}.json'
    summary = {'dataset': ds.key, 'variant': args.variant, 'folds': {}}
    from cirf.evaluate import evaluate_fold
    for k, d in enumerate(dirs):
        t0 = time.time()
        res = evaluate_fold(model_type, d, ds=ds, device=device, verbose=False)
        npz = np.load(d / 'eval' / 'eval_data.npz')
        entry = {'ev': np.asarray(res['ev']).tolist(),
                 'adj_r2': np.asarray(res['adj_r2']).tolist(),
                 'cc_norm': cc_norm_per_neuron(npz['y_pred'], npz['test_rep']).tolist()
                 if 'test_rep' in npz.files else None,
                 'lsta_corr': np.asarray(res['lsta_corr']).tolist()}
        if model_type == 'transformer':
            entry['centres'] = centre_stats(d, args.variant, ds, device)
        summary['folds'][k] = entry
        c = entry.get('centres')
        print(f'{ds.key} {args.variant} fold {k}: EV {np.mean(entry["ev"]):.4f} '
              f'LSTA {np.mean(entry["lsta_corr"]):.4f}'
              + (f'  centres moved {np.median(c["moved_px"]):.2f} px (median), '
                 f'to STA {np.median(c["dist_to_sta_trained_px"]):.2f} px'
                 if c else '')
              + f'  ({time.time() - t0:.0f}s)', flush=True)
        out.write_text(json.dumps(summary), encoding='utf-8')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
