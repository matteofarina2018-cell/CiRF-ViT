"""Fit the per-neuron LN models.

Each neuron gets `N_REPEATS` independent fits on its own RF window; they differ
in initialisation and batch order, and their predictions are averaged. One call
fits one fold of the K-fold cross-validation (`cirf.data.kfold_split`): the
held-out fold is the validation set of every repeat. The checkpoints mirror
the GP's per-cell layout:

    <out_dir>/config.json
    <out_dir>/cell_XX/seed_K.pt

Fitting is cheap (a few thousand parameters per neuron) and runs fine on CPU,
which is the default so a GPU stays free for the shared-core models.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cirf import config as cfg_mod
from cirf import data as dat

from .config import get_cfg
from .losses import poisson_loss, filter_regularization
from .model import LNModel


def train_cell(Xtr, Ytr, Xva, Yva, mean_rate, cfg, device, seed):
    """Fit one LN model; returns (model, best val loss, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LNModel(cfg['CROP_SIZE'], mean_rate=mean_rate).to(device)
    opt = Adam(model.parameters(), lr=cfg['LEARNING_RATE'])
    sched = ReduceLROnPlateau(opt, 'min', factor=cfg['LR_DECAY'],
                              patience=cfg['LR_PATIENCE'])
    best, best_state, patience = float('inf'), copy.deepcopy(model.state_dict()), 0
    hist = {'train_loss': [], 'val_loss': []}
    n, bs = Xtr.shape[0], cfg['BATCH_SIZE']

    for _epoch in range(1, cfg['MAX_EPOCHS'] + 1):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = poisson_loss(model(Xtr[idx]), Ytr[idx]) + filter_regularization(
                model.filter, cfg['SPARSITY_FACTOR'], cfg['SMOOTHNESS_FACTOR'])
            loss.backward()
            opt.step()
            model.clamp_filter_norm(cfg['MAX_NORM'])
        model.eval()
        with torch.no_grad():
            tp = poisson_loss(model(Xtr), Ytr).item()
            vp = poisson_loss(model(Xva), Yva).item()
        sched.step(vp)
        hist['train_loss'].append(tp)
        hist['val_loss'].append(vp)
        if vp < best - 1e-6:
            best, best_state, patience = vp, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= cfg['EARLY_STOP']:
                break

    model.load_state_dict(best_state)
    return model, best, hist


def train(ds=None, cfg=None, out_dir=None, device='cpu', cells=None,
          n_repeats=None, fold=0, n_folds=None, data_fraction=1.0, seed=None,
          save=True, verbose=True):
    """Fit every neuron on one fold; returns {'val_loss': (N, K), 'out_dir'}.

    ``fold`` / ``n_folds`` pick the held-out fold of the K-fold cross-validation;
    ``data_fraction`` and ``seed`` subset the training folds the way the other
    models do (see `cirf.data.fold_indices`).
    """
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    cfg = dict(cfg or get_cfg(dataset))
    device = torch.device(device or 'cpu')
    n_folds = int(n_folds or cfg_mod.N_FOLDS)
    fold = int(fold)
    out_dir = Path(out_dir or cfg_mod.fold_dir('LN', fold, dataset))
    cfg['CROP_SIZE'] = int(min(cfg['CROP_SIZE'], dat.img_size(dataset)))
    K = int(n_repeats or cfg['N_REPEATS'])
    cs = cfg['CROP_SIZE']

    N = dat.n_cells(dataset)
    cells = list(range(N)) if cells is None else list(cells)
    cx_all, cy_all = dat.centers(dataset)
    pool_imgs, pool_resp = dat.images('pool', dataset), dat.responses('pool', dataset)
    i_tr, i_vl = dat.fold_indices(dataset, fold, n_folds, data_fraction,
                                  42 if seed is None else seed)
    images = {'train': pool_imgs[i_tr], 'val': pool_imgs[i_vl]}
    resp = {'train': pool_resp[i_tr], 'val': pool_resp[i_vl]}

    if save:
        cfg_mod.ensure(out_dir)
        (out_dir / 'config.json').write_text(json.dumps(
            {'cfg': cfg, 'dataset': dataset.key, 'n_cells': N, 'seed': seed,
             'data_fraction': data_fraction, 'fold': fold, 'n_folds': n_folds,
             'val_indices': [int(i) for i in i_vl]}, indent=2),
            encoding='utf-8')
    if verbose:
        print(f'LN on {dataset.label}: {len(cells)} neurons x {K} repeats, '
              f'{cs} px window, fold {fold}/{n_folds} '
              f'({len(i_tr)} train / {len(i_vl)} held out)')

    t0 = time.time()
    val = np.zeros((len(cells), K), dtype=np.float32)
    for i, c in enumerate(cells):
        cxc, cyc = float(cx_all[c]), float(cy_all[c])
        Xtr = torch.tensor(dat.crop_around_center(images['train'], cxc, cyc, cs), device=device)
        Xva = torch.tensor(dat.crop_around_center(images['val'], cxc, cyc, cs), device=device)
        Ytr = torch.tensor(resp['train'][:, c], dtype=torch.float32, device=device)
        Yva = torch.tensor(resp['val'][:, c], dtype=torch.float32, device=device)
        mean_rate = float(resp['train'][:, c].mean())

        cell_dir = cfg_mod.ensure(out_dir / f'cell_{c:02d}') if save else None
        for k in range(K):
            model, best, hist = train_cell(Xtr, Ytr, Xva, Yva, mean_rate, cfg,
                                           device, k + 1000 * (seed or 0))
            val[i, k] = best
            if save:
                torch.save({'state_dict': {kk: v.cpu() for kk, v in model.state_dict().items()},
                            'val_loss': best, 'history': hist,
                            'center': (cxc, cyc), 'crop_size': cs,
                            'cell': c, 'seed': k},
                           cell_dir / f'seed_{k}.pt')
        if verbose and (i == 0 or (i + 1) % 10 == 0 or i == len(cells) - 1):
            print(f'  cell {c:02d}  val {val[i].mean():.4f}  '
                  f'({time.time() - t0:.0f}s elapsed)', flush=True)

    if verbose:
        print(f'  saved -> {out_dir}  ({time.time() - t0:.0f}s)')
    return {'val_loss': val, 'best_val': float(val.mean()), 'out_dir': out_dir,
            'cfg': cfg, 'fold': fold, 'n_folds': n_folds}
