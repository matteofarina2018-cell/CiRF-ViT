"""Benchmarks: predictions, metrics and the LSTA comparison, for every model.

Every model is trained by K-fold cross-validation, so a model is K fold
checkpoints. `evaluate_fold` turns one of them into a ``fold_K/eval/eval_data.npz``
holding its test predictions, per-neuron metrics, its out-of-fold predictions
on the pool stimuli it never saw, and (optionally) the input gradients the LSTA
comparison scores. `evaluate_model` runs it for every fold and writes the
aggregated ``eval/eval_data.npz``: per-neuron metrics averaged over the folds
(with every fold kept alongside), the fold-ensemble prediction, and the
cross-validated score assembled from the out-of-fold predictions of the whole
pool. Every downstream figure reads those files and never a checkpoint, so
plotting is cheap and never re-runs a model.

`fraction_metrics` is the data-efficiency benchmark: it re-scores the models
trained on a fraction of the data (see scripts/fraction_study.py) and returns
per-neuron metrics for every (model, fraction, run), a run being one fold and
one seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import checkpoints as ckpt
from . import config as cfg_mod
from . import data as dat
from . import lsta as lsta_mod
from .metrics import (adjusted_r2_per_neuron, explained_variance_per_neuron,
                      METRIC_FNS)


# -----------------------------------------------------------------------------
# Predictions
# -----------------------------------------------------------------------------

def predict_torch(model, images, device='cpu', batch_size=64):
    """(n_img, N) predicted rates from a pixel-space model."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = torch.tensor(images[i:i + batch_size, np.newaxis],
                             dtype=torch.float32, device=device)
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_gp(models, predict_fn, images, device='cpu', ds=None):
    """(n_img, N) predicted rates from the per-cell GPs."""
    crop, off = dat.gp_crop(ds), dat.gp_offset(ds)
    patch = images[:, off:off + crop, off:off + crop]
    X = torch.tensor(patch.reshape(patch.shape[0], -1),
                     dtype=torch.float32, device=device)
    out = np.zeros((X.shape[0], len(models)), dtype=np.float32)
    for n, m in enumerate(models):
        out[:, n] = predict_fn(m, X)['f_pred'].cpu().numpy()
    return out


def predict_ln(cells, images, crop_size, ds=None, device='cpu', per_seed=False):
    """(n_img, N) predicted rates from the per-cell LN repeats.

    Each neuron's prediction is the mean over its repeats, or all of them
    stacked as (n_seeds, n_img, N) with `per_seed=True`.
    """
    cx_all, cy_all = dat.centers(ds)
    n_seeds = max(len(s) for s in cells)
    out = np.zeros((n_seeds, len(images), len(cells)), dtype=np.float32)
    for c, seeds in enumerate(cells):
        X = torch.tensor(dat.crop_around_center(images, cx_all[c], cy_all[c], crop_size),
                         dtype=torch.float32, device=device)
        with torch.no_grad():
            for k, m in enumerate(seeds):
                out[k, :, c] = m(X).cpu().numpy()
    return out if per_seed else out.mean(axis=0)


def predict(model_type, model_dir=None, ds=None, device='cpu', images=None):
    """(n_img, N) predictions of a trained checkpoint on the test stimuli."""
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    if images is None:
        images = dat.images('test', dataset)
    if model_type == 'GP':
        models, fn = ckpt.load_gp_models(model_dir, dataset, device)
        return predict_gp(models, fn, images, device, dataset)
    if model_type == 'LN':
        cells, cfg = ckpt.load_ln_models(model_dir, dataset, device)
        return predict_ln(cells, images, int(cfg['CROP_SIZE']), dataset, device)
    model, _ = ckpt.load_torch_model(model_type, model_dir, dataset, device)
    return predict_torch(model, images, device)


# -----------------------------------------------------------------------------
# Out-of-fold scores on the pool
#
# The pool responses are single presentations, so the reliability-normalised
# metrics of `cirf.metrics` do not apply to them; what a held-out fold gives is
# the classic cross-validated score of a Poisson model: the Poisson negative
# log-likelihood (the training objective) and the correlation with the
# single-trial spike counts, both per neuron.
# -----------------------------------------------------------------------------

def poisson_nll_per_neuron(y_pred, y_true, eps=1e-8):
    """(N,) mean Poisson NLL of predicted rates against spike counts."""
    rate = np.clip(np.asarray(y_pred, dtype=np.float64), eps, None)
    y = np.asarray(y_true, dtype=np.float64)
    return (rate - y * np.log(rate)).mean(axis=0)


def corr_per_neuron(y_pred, y_true):
    """(N,) Pearson correlation of predicted rates with single-trial counts.

    A constant prediction (or response) has no correlation and scores 0.
    """
    p = np.asarray(y_pred, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    p = p - p.mean(axis=0)
    y = y - y.mean(axis=0)
    denom = np.sqrt((p ** 2).sum(axis=0) * (y ** 2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (p * y).sum(axis=0) / denom
    return np.where(np.isfinite(r), r, 0.0)


# -----------------------------------------------------------------------------
# Full evaluation of one fold checkpoint
# -----------------------------------------------------------------------------

def evaluate_fold(model_type, model_dir, out_dir=None, ds=None, device=None,
                  compute_lsta=True, verbose=True):
    """Score one fold checkpoint and write its ``eval_data.npz``.

    Stored: y_pred, y_true_mean, test_rep, explained_variance, adjusted_r2,
    lsta_corr, the training history, the out-of-fold predictions on the
    held-out pool stimuli (`oof_pred`, `oof_indices`) with their Poisson NLL
    and correlation per neuron (`cv_poisson_nll`, `cv_corr`), and (when
    `compute_lsta`) the model's input-gradient maps `lsta_model`, which the
    LSTA figures reuse.

    The LSTA part is skipped automatically when the dataset carries no
    experimental LSTA -- prediction metrics work on any recording.
    """
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model_dir = Path(model_dir)
    out_dir = cfg_mod.ensure(out_dir or (model_dir / 'eval'))
    fold, n_folds = ckpt.run_fold(model_dir)
    oof_idx = ckpt.held_out_indices(model_dir, dataset)

    images_test = dat.images('test', dataset)
    test_rep = dat.test_repeats(dataset)
    y_true_mean = test_rep.mean(axis=1)
    images_oof = dat.images('pool', dataset)[oof_idx]
    resp_oof = dat.responses('pool', dataset)[oof_idx]

    want_lsta = compute_lsta and dat.meta(dataset)['has_lsta']
    base_imgs = dat.lsta_base_images(dataset) if want_lsta else None
    ref = dat.lsta_reference(dataset) if want_lsta else None
    maps = None

    if model_type == 'GP':
        models, fn = ckpt.load_gp_models(model_dir, dataset, device)
        y_pred = predict_gp(models, fn, images_test, device, dataset)
        oof_pred = predict_gp(models, fn, images_oof, device, dataset)
        if want_lsta:
            maps = lsta_mod.gp_gradients(models, base_imgs, device, dataset)
    elif model_type == 'LN':
        cells, cfg = ckpt.load_ln_models(model_dir, dataset, device)
        crop = int(cfg['CROP_SIZE'])
        y_pred = predict_ln(cells, images_test, crop, dataset, device)
        oof_pred = predict_ln(cells, images_oof, crop, dataset, device)
        if want_lsta:
            maps = lsta_mod.ln_gradients(cells, base_imgs, crop, dataset, device)
    else:
        model, _cfg = ckpt.load_torch_model(model_type, model_dir, dataset, device)
        y_pred = predict_torch(model, images_test, device)
        oof_pred = predict_torch(model, images_oof, device)
        if want_lsta:
            maps = lsta_mod.gradients(model_type, model, base_imgs, device,
                                      n_cells=dat.n_cells(dataset), ds=dataset)

    ev = explained_variance_per_neuron(y_pred, test_rep)
    adj = adjusted_r2_per_neuron(y_pred, test_rep)
    lsta_corr = np.zeros(y_pred.shape[1], dtype=np.float32)
    if maps is not None:
        lsta_corr = lsta_mod.corr_from_arrays(maps, ref['lsta'], ref['ellipses'],
                                              maps.shape[-1], ds=dataset)
    cv_nll = poisson_nll_per_neuron(oof_pred, resp_oof)
    cv_corr = corr_per_neuron(oof_pred, resp_oof)

    hist = {}
    hist_path = model_dir / 'history.json'
    if hist_path.exists():
        import json
        hist = json.loads(hist_path.read_text(encoding='utf-8'))

    payload = dict(
        y_pred=y_pred.astype(np.float32),
        y_true_mean=y_true_mean.astype(np.float32),
        test_rep=test_rep.astype(np.float32),
        images_test=images_test.astype(np.float32),
        explained_variance=ev.astype(np.float32),
        adjusted_r2=adj.astype(np.float32),
        lsta_corr=lsta_corr.astype(np.float32),
        oof_pred=oof_pred.astype(np.float32),
        oof_indices=np.asarray(oof_idx, dtype=np.int64),
        cv_poisson_nll=cv_nll.astype(np.float32),
        cv_corr=cv_corr.astype(np.float32),
        train_loss=np.array(hist.get('train_loss', []), dtype=np.float32),
        val_loss=np.array(hist.get('val_loss', []), dtype=np.float32),
        fold=np.int32(fold),
        n_folds=np.int32(n_folds),
        n_cells=np.int32(y_pred.shape[1]),
        model_type=np.array(model_type),
        dataset=np.array(dataset.key),
    )
    if maps is not None:
        payload['lsta_model'] = maps.astype(np.float32)
        payload['lsta_image_indices'] = np.asarray(ref['image_indices'], dtype=np.int64)

    np.savez(out_dir / 'eval_data.npz', **payload)
    if verbose:
        print(f'  {model_type:>11} fold {fold}: EV={ev.mean():.4f}  '
              f'adjR2={adj.mean():.4f}  LSTA={lsta_corr.mean():.4f}  '
              f'CV-NLL={cv_nll.mean():.4f}   -> {out_dir / "eval_data.npz"}')
    return {'y_pred': y_pred, 'y_true_mean': y_true_mean, 'test_rep': test_rep,
            'ev': ev, 'adj_r2': adj, 'lsta_corr': lsta_corr, 'lsta_model': maps,
            'lsta_image_indices': None if ref is None else ref['image_indices'],
            'oof_pred': oof_pred, 'oof_indices': oof_idx, 'cv_poisson_nll': cv_nll,
            'cv_corr': cv_corr, 'history': hist, 'fold': fold, 'n_folds': n_folds}


# -----------------------------------------------------------------------------
# Aggregation over the folds
# -----------------------------------------------------------------------------

def _pad_stack(rows):
    """(n_rows, max_len) float32, NaN-padded, from rows of unequal length."""
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def aggregate_folds(model_type, fold_results, ds=None, out_path=None,
                    verbose=True):
    """Combine the per-fold evaluations into the model's ``eval_data.npz``.

    ``fold_results`` maps fold -> the dict `evaluate_fold` returns (or the
    dict a stored fold ``eval_data.npz`` loads into). Stored:

        explained_variance, adjusted_r2, lsta_corr    (N,)  mean over the folds
        <metric>_folds                                (K, N) every fold
        y_pred                       (n_test, N) mean prediction of the K fold
                                     models (the fold ensemble); y_pred_folds
                                     holds each fold's
        explained_variance_ensemble, adjusted_r2_ensemble    (N,) of y_pred
        lsta_model                   mean input-gradient map over the folds
                                     (per-fold maps stay in fold_K/eval/)
        oof_pred                     (n_pool, N) out-of-fold prediction of every
                                     pool stimulus, from the one fold model that
                                     never saw it; NaN where the fold is missing
        oof_fold                     (n_pool,) which fold predicted it, -1 missing
        cv_poisson_nll, cv_corr      (N,) the cross-validated score over the
                                     whole pool
        train_loss, val_loss         (max_epochs,) mean learning curve over the
                                     folds; <loss>_folds the NaN-padded curves
        folds, n_folds
    """
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    out_path = Path(out_path or cfg_mod.eval_path(model_type, dataset))
    folds = sorted(fold_results)
    if not folds:
        raise ValueError(f'no fold evaluation of {model_type} to aggregate')
    first = fold_results[folds[0]]

    def key(r, name, alt=None):
        return r[name] if name in r else r[alt]

    ev_f = np.stack([key(fold_results[k], 'explained_variance', 'ev') for k in folds])
    adj_f = np.stack([key(fold_results[k], 'adjusted_r2', 'adj_r2') for k in folds])
    lsta_f = np.stack([fold_results[k]['lsta_corr'] for k in folds])
    y_f = np.stack([fold_results[k]['y_pred'] for k in folds])
    nll_f = np.stack([fold_results[k]['cv_poisson_nll'] for k in folds])
    corr_f = np.stack([fold_results[k]['cv_corr'] for k in folds])

    test_rep = first['test_rep']
    y_ens = y_f.mean(axis=0)
    n_pool = dat.meta(dataset)['n_pool']
    n_cells = y_ens.shape[1]
    oof = np.full((n_pool, n_cells), np.nan, dtype=np.float32)
    oof_fold = np.full(n_pool, -1, dtype=np.int64)
    for k in folds:
        idx = np.asarray(fold_results[k]['oof_indices'], dtype=np.int64)
        oof[idx] = fold_results[k]['oof_pred']
        oof_fold[idx] = k
    covered = oof_fold >= 0
    resp_pool = dat.responses('pool', dataset)
    cv_nll = poisson_nll_per_neuron(oof[covered], resp_pool[covered])
    cv_corr = corr_per_neuron(oof[covered], resp_pool[covered])

    tr_f = _pad_stack([np.asarray(fold_results[k].get('train_loss',
                       fold_results[k].get('history', {}).get('train_loss', [])),
                       dtype=np.float32) for k in folds])
    vl_f = _pad_stack([np.asarray(fold_results[k].get('val_loss',
                       fold_results[k].get('history', {}).get('val_loss', [])),
                       dtype=np.float32) for k in folds])
    with np.errstate(all='ignore'):
        tr_mean = np.nanmean(tr_f, axis=0) if tr_f.size else np.zeros(0, np.float32)
        vl_mean = np.nanmean(vl_f, axis=0) if vl_f.size else np.zeros(0, np.float32)

    payload = dict(
        y_pred=y_ens.astype(np.float32),
        y_pred_folds=y_f.astype(np.float32),
        y_true_mean=first['y_true_mean'].astype(np.float32),
        test_rep=test_rep.astype(np.float32),
        images_test=dat.images('test', dataset).astype(np.float32),
        explained_variance=ev_f.mean(axis=0).astype(np.float32),
        explained_variance_folds=ev_f.astype(np.float32),
        explained_variance_ensemble=explained_variance_per_neuron(y_ens, test_rep).astype(np.float32),
        adjusted_r2=adj_f.mean(axis=0).astype(np.float32),
        adjusted_r2_folds=adj_f.astype(np.float32),
        adjusted_r2_ensemble=adjusted_r2_per_neuron(y_ens, test_rep).astype(np.float32),
        lsta_corr=lsta_f.mean(axis=0).astype(np.float32),
        lsta_corr_folds=lsta_f.astype(np.float32),
        oof_pred=oof, oof_fold=oof_fold,
        cv_poisson_nll=cv_nll.astype(np.float32),
        cv_poisson_nll_folds=nll_f.astype(np.float32),
        cv_corr=cv_corr.astype(np.float32),
        cv_corr_folds=corr_f.astype(np.float32),
        train_loss=tr_mean.astype(np.float32), train_loss_folds=tr_f,
        val_loss=vl_mean.astype(np.float32), val_loss_folds=vl_f,
        folds=np.asarray(folds, dtype=np.int64),
        n_folds=np.int32(first['n_folds']),
        n_cells=np.int32(n_cells),
        model_type=np.array(model_type),
        dataset=np.array(dataset.key),
    )
    maps = [fold_results[k].get('lsta_model') for k in folds]
    if all(m is not None for m in maps):
        payload['lsta_model'] = np.mean(np.stack(maps), axis=0).astype(np.float32)
        idx = first.get('lsta_image_indices')
        if idx is None:
            idx = dat.lsta_reference(dataset)['image_indices']
        payload['lsta_image_indices'] = np.asarray(idx, dtype=np.int64)

    cfg_mod.ensure(out_path.parent)
    np.savez(out_path, **payload)
    if verbose:
        sd = ev_f.mean(axis=1).std()
        print(f'  {model_type:>11} {len(folds)}-fold: '
              f'EV={payload["explained_variance"].mean():.4f}±{sd:.4f}  '
              f'adjR2={payload["adjusted_r2"].mean():.4f}  '
              f'LSTA={payload["lsta_corr"].mean():.4f}  '
              f'CV-NLL={cv_nll.mean():.4f}  CV-r={cv_corr.mean():.4f}'
              + ('' if covered.all() else
                 f'  [{int((~covered).sum())} pool stimuli without a fold]')
              + f'   -> {out_path}')
    return payload


def evaluate_model(model_type, model_dir=None, out_dir=None, ds=None,
                   device=None, compute_lsta=True, folds=None, verbose=True):
    """Score every trained fold of a model and write the aggregated ``eval_data.npz``.

    ``model_dir`` is the root holding the ``fold_K/`` directories (default:
    the model's checkpoint directory); ``folds`` restricts which of them are
    scored (default: every trained one). Each fold's own evaluation lands in
    ``fold_K/eval/``, the aggregate in ``out_dir`` (default ``<root>/eval/``).
    Returns the aggregated payload (see `aggregate_folds`).
    """
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    root = Path(model_dir or cfg_mod.checkpoint_dir(model_type, dataset))
    trained = dict(ckpt.fold_dirs(model_type, dataset, root))
    wanted = sorted(trained) if folds is None else [int(k) for k in folds]
    missing = [k for k in wanted if k not in trained]
    if missing or not wanted:
        raise FileNotFoundError(
            f'no trained fold{"s " + str(missing) if missing else ""} of '
            f'{model_type} under {root} (scripts/train.py --model {model_type})')
    results = {}
    for k in wanted:
        results[k] = evaluate_fold(model_type, trained[k], ds=dataset,
                                   device=device, compute_lsta=compute_lsta,
                                   verbose=verbose)
    out_path = (Path(out_dir) / 'eval_data.npz') if out_dir else \
        (root / 'eval' / 'eval_data.npz')
    return aggregate_folds(model_type, results, dataset, out_path, verbose)


def load_eval(model_type, ds=None, path=None):
    """The aggregated ``eval_data.npz`` as a dict, or None when it is missing."""
    p = Path(path or cfg_mod.eval_path(model_type, ds))
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def load_fold_evals(model_type, ds=None, root=None) -> dict:
    """{fold: dict} of every stored per-fold ``eval_data.npz``."""
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    out = {}
    for k, d in ckpt.fold_dirs(model_type, dataset, root):
        p = Path(d) / 'eval' / 'eval_data.npz'
        if p.exists():
            out[k] = load_eval(model_type, dataset, p)
    return out


# -----------------------------------------------------------------------------
# Data-efficiency benchmark
# -----------------------------------------------------------------------------

def fraction_metrics(model_types=('GP', 'cnn', 'transformer'), fractions=None,
                     folds=None, seeds=None, metric='adj_r2', ds=None,
                     device=None, verbose=True):
    """Per-neuron metrics of every fraction-study run.

    Returns ``results[model][fraction][(fold, seed)] = (N,) metric``, rebuilt
    from the checkpoints under ``results/<dataset>/fraction_study/``. A run is
    one fold of the cross-validation and one training seed. Runs that are
    missing are skipped, so a partial sweep still plots.
    """
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    fractions = list(fractions or cfg_mod.FRACTIONS)
    folds = list(folds if folds is not None else cfg_mod.FOLDS)
    seeds = list(seeds if seeds is not None else cfg_mod.FRAC_SEEDS)
    metric_fn = METRIC_FNS[metric] if isinstance(metric, str) else metric

    images_test = dat.images('test', dataset)
    test_rep = dat.test_repeats(dataset)

    results = {}
    for model_type in model_types:
        per_frac = {}
        for f in fractions:
            per_run = {}
            for k in folds:
                for s in seeds:
                    d = cfg_mod.fraction_dir(model_type, f, k, s, dataset)
                    if not ckpt.exists(model_type, d):
                        continue
                    try:
                        y_pred = predict(model_type, d, dataset, device, images_test)
                    except Exception as exc:  # a broken run must not kill the sweep
                        if verbose:
                            print(f'  [skip] {model_type} frac={f:.0%} fold={k} '
                                  f'seed={s}: {exc}')
                        continue
                    per_run[(k, s)] = metric_fn(y_pred, test_rep)
            if per_run:
                per_frac[f] = per_run
                if verbose:
                    mu = np.mean([v.mean() for v in per_run.values()])
                    print(f'  {model_type:>11} frac {f:>4.0%}  '
                          f'{len(per_run)} runs  mean {metric} = {mu:.4f}')
        if per_frac:
            results[model_type] = per_frac
    return results


def flatten_fraction_results(results) -> dict:
    """results[model][frac][(fold, seed)] -> a flat dict of arrays, for np.savez."""
    return {f'{m}|{f:.2f}|{k}|{s}': np.asarray(a, dtype=np.float32)
            for m, per_frac in results.items()
            for f, per_run in per_frac.items()
            for (k, s), a in per_run.items()}


def unflatten_fraction_results(flat) -> dict:
    """Inverse of `flatten_fraction_results`."""
    out = {}
    for key, arr in flat.items():
        parts = key.split('|')
        if len(parts) != 4:
            raise ValueError(
                f'{key!r}: this cache predates the K-fold cross-validation '
                f'(runs were keyed by seed only). Delete it or pass --refresh.')
        m, f, k, s = parts
        out.setdefault(m, {}).setdefault(float(f), {})[(int(k), int(s))] = np.asarray(arr)
    return out


# -----------------------------------------------------------------------------
# Summarising a fraction curve
#
# The sweep trains one run per (fold, seed) at every point and every one of
# them is used: there is no run selection anywhere in this repository, and no
# figure drops a run. A curve is the mean over all the runs that exist at that
# point, and its band is their spread.
# -----------------------------------------------------------------------------

def curve(results, model_type, ds=None):
    """(x %, mean, spread) of a fraction curve, ready to plot.

    The spread is the mean across runs of the metric's standard deviation over
    neurons -- the band drawn in the paper's figure.
    """
    xs, means, bands = [], [], []
    for f in sorted(results.get(model_type, {})):
        per_run = results[model_type][f]
        if not per_run:
            continue
        stack = np.stack([per_run[r] for r in sorted(per_run)], axis=0)
        xs.append(f * 100)
        means.append(float(stack.mean(axis=1).mean()))
        bands.append(float(stack.std(axis=0).mean()))
    return np.asarray(xs), np.asarray(means), np.asarray(bands)


def per_neuron_at(results, model_type, frac=1.0, ds=None):
    """(N,) metric per neuron at one fraction, averaged over every run."""
    runs = per_run_at(results, model_type, frac, ds)
    return None if runs is None else runs.mean(axis=0)


def per_run_at(results, model_type, frac=1.0, ds=None):
    """(n_runs, N) metric per neuron, one row per run."""
    per_run = results.get(model_type, {}).get(frac, {})
    if not per_run:
        return None
    return np.stack([per_run[r] for r in sorted(per_run)], axis=0)
