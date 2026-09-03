#!/usr/bin/env python
"""Score trained checkpoints: predictions, metrics, LSTA comparison.

    python scripts/evaluate.py                          # every model, mouse
    python scripts/evaluate.py --dataset both
    python scripts/evaluate.py --model transformer --data my.npz
    python scripts/evaluate.py --no-lsta                # metrics only, faster

Every model is K fold checkpoints (scripts/train.py). Each fold is scored on
its own into results/<dataset>/checkpoints/<model>/fold_K/eval/eval_data.npz,
then the folds are aggregated into results/<dataset>/checkpoints/<model>/eval/
eval_data.npz -- per-neuron metrics averaged over the folds, with every fold
kept alongside, the fold-ensemble prediction, and the cross-validated score
assembled from the out-of-fold predictions of the whole stimulus pool. Every
figure script reads the aggregated file, so this is the step between training
and plotting. The summary table reports the mean over folds and, in
parentheses, the standard deviation across folds of the per-fold mean.

Metrics (per neuron, against the repeated test responses):
    EV        explained variance, normalised by the neuron's own reliability
    adj R^2   the squared version of the same quantity
    LSTA r    correlation between the model's input gradient and the measured
              local spike-triggered average, band-limited to the frequencies
              the experiment resolves. Only for datasets that carry an LSTA.
Cross-validated, on the pool stimuli each fold never saw (single trials):
    CV NLL    Poisson negative log-likelihood, the training objective
    CV r      correlation with the single-trial spike counts
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import (add_dataset_arg, add_device_arg, banner, cfg,
                     datasets_from, device_from)
from cirf import checkpoints as ckpt
from cirf import evaluate as ev

MODELS = ('LN', 'GP', 'cnn', 'transformer')


def _row(m, e):
    """(label, EV, adjR2, LSTA, CV-NLL, CV-r, sd of the per-fold mean EV)."""
    ev_folds = np.asarray(e.get('explained_variance_folds',
                                e['explained_variance'][None]))
    return (m, float(np.mean(e['explained_variance'])),
            float(np.mean(e['adjusted_r2'])), float(np.mean(e['lsta_corr'])),
            float(np.mean(e['cv_poisson_nll'])) if 'cv_poisson_nll' in e else np.nan,
            float(np.mean(e['cv_corr'])) if 'cv_corr' in e else np.nan,
            float(ev_folds.mean(axis=1).std()), int(ev_folds.shape[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='all', choices=(*MODELS, 'all'))
    add_dataset_arg(ap)
    add_device_arg(ap)
    ap.add_argument('--folds', type=int, nargs='+', default=None,
                    help='score only these folds (default: every trained fold)')
    ap.add_argument('--no-lsta', action='store_true',
                    help='skip the input-gradient maps (much faster)')
    ap.add_argument('--force', action='store_true',
                    help='re-score models that already have an eval_data.npz')
    args = ap.parse_args()

    device = device_from(args)
    models = MODELS if args.model == 'all' else (args.model,)

    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        banner(f'evaluate | {ds.label}')
        rows = []
        for m in models:
            if not ckpt.exists(m, ds=ds):
                print(f'  [skip] {m}: no fold checkpoint in {cfg.checkpoint_dir(m, ds)}')
                continue
            trained = ckpt.folds(m, ds)
            if cfg.eval_path(m, ds).exists() and not args.force:
                res = ev.load_eval(m, ds)
                have = list(np.asarray(res.get('folds', [])))
                if sorted(have) == sorted(trained) and args.folds is None:
                    print(f'  [keep] {m}: eval_data.npz covers folds {have} '
                          f'(--force to redo)')
                    rows.append(_row(m, res))
                    continue
                print(f'  [redo] {m}: eval_data.npz covers folds {have}, '
                      f'trained folds are {trained}')
            try:
                res = ev.evaluate_model(m, ds=ds, device=device, folds=args.folds,
                                        compute_lsta=not args.no_lsta)
                rows.append(_row(m, res))
            except Exception as exc:
                print(f'  [error] {m}: {exc}')

        if rows:
            print(f'\n  {"model":>12}  {"folds":>5}  {"EV":>14}  {"adj R2":>8}  '
                  f'{"LSTA r":>8}  {"CV NLL":>8}  {"CV r":>8}')
            for m, e, a, l, nll, r, sd, k in rows:
                print(f'  {cfg.MODEL_LABELS[m]:>12}  {k:>5d}  '
                      f'{e:8.4f} ({sd:.3f})  {a:8.4f}  {l:8.4f}  {nll:8.4f}  {r:8.4f}')


if __name__ == '__main__':
    main()
