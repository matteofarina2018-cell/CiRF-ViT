"""Build and train the CNN baseline on any dataset.

Adam on the regularised Poisson objective, ReduceLROnPlateau, early stopping
with best-weight restore, and a projection that keeps the readout non-negative
after every step. The whole training set is held on the device: these
recordings are a few thousand small images, so batching from host memory would
dominate the runtime. One call fits one fold of the K-fold cross-validation
(`cirf.data.kfold_split`): the held-out fold is the validation set.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cirf import config as cfg_mod
from cirf import data as dat

from .config import get_cfg
from .losses import poisson_loss, total_loss
from .model import RetinalModel


def build_model(cfg: dict, ds=None, device='cpu') -> RetinalModel:
    """Instantiate the CNN for a dataset, reading geometry from the data."""
    cx, cy = dat.centers(ds)
    cfg = dict(cfg)
    cfg.setdefault('IMG_SIZE', dat.img_size(ds))
    cfg.setdefault('N_CELLS', dat.n_cells(ds))
    return RetinalModel(cfg, cx, cy).to(device)


def train(ds=None, cfg=None, out_dir=None, seed=None, data_fraction=1.0,
          device=None, max_epochs=None, fold=0, n_folds=None, save=True,
          verbose=True):
    """Train one fold; returns {'best_val', 'history', 'n_params', 'out_dir'}.

    ``fold`` / ``n_folds`` pick the held-out fold of the K-fold cross-validation
    (see `cirf.data.fold_indices` for how ``data_fraction`` and ``seed`` act).
    """
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    cfg = dict(cfg or get_cfg(dataset))
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    n_folds = int(n_folds or cfg_mod.N_FOLDS)
    fold = int(fold)
    out_dir = Path(out_dir or cfg_mod.fold_dir('cnn', fold, dataset))
    if max_epochs:
        cfg['MAX_EPOCHS'] = int(max_epochs)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    images_pool = dat.images('pool', dataset)
    responses_pool = dat.responses('pool', dataset)
    i_tr, i_vl = dat.fold_indices(dataset, fold, n_folds, data_fraction,
                                  42 if seed is None else seed)
    images_train, responses_train = images_pool[i_tr], responses_pool[i_tr]
    images_val, responses_val = images_pool[i_vl], responses_pool[i_vl]

    Xtr = torch.tensor(images_train[:, None], dtype=torch.float32, device=device)
    Ytr = torch.tensor(responses_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(images_val[:, None], dtype=torch.float32, device=device)
    Yva = torch.tensor(responses_val, dtype=torch.float32, device=device)
    n, bs = Xtr.shape[0], cfg['BATCH_SIZE']

    model = build_model(cfg, dataset, device)
    # start each neuron at its mean firing rate: softplus(bias) = mean rate
    mean_rate = torch.tensor(responses_train.mean(0), dtype=torch.float32)
    model.readout.bias.data.copy_(
        torch.log(torch.expm1(torch.clamp(mean_rate, min=1e-3))).to(device))

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f'CNN on {dataset.label}: {model.n_cells} neurons, {n_params:,} '
              f'parameters, feature map {model.feat}x{model.feat}, '
              f'fold {fold}/{n_folds} ({len(i_tr)} train / {len(i_vl)} held out), '
              f'device {device}')

    opt = Adam(model.parameters(), lr=cfg['LEARNING_RATE'])
    sched = ReduceLROnPlateau(opt, 'min', factor=cfg['LR_DECAY'],
                              patience=cfg['LR_PATIENCE'])

    best, best_state, patience = float('inf'), copy.deepcopy(model.state_dict()), 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, cfg['MAX_EPOCHS'] + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tr = 0.0
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            opt.zero_grad()
            loss, pois = total_loss(model(Xtr[idx]), Ytr[idx], model, cfg)
            loss.backward()
            opt.step()
            model.clamp_nonneg()
            tr += pois.item()
        tr /= (n // bs + 1)

        model.eval()
        with torch.no_grad():
            vp = poisson_loss(model(Xva), Yva).item()
        sched.step(vp)

        history['train_loss'].append(float(tr))
        history['val_loss'].append(float(vp))

        improved = vp < best - 1e-6
        if improved:
            best, best_state, patience = vp, copy.deepcopy(model.state_dict()), 0
            history['best_epoch'] = epoch
        else:
            patience += 1
        if verbose and (improved or epoch % 10 == 0):
            print(f'  epoch {epoch:4d}  train {tr:.4f}  val {vp:.4f}'
                  f'{"  *" if improved else ""}')
        if patience >= cfg['EARLY_STOP']:
            if verbose:
                print(f'  early stop at epoch {epoch} (best val {best:.4f})')
            break

    model.load_state_dict(best_state)

    if save:
        cfg_mod.ensure(out_dir)
        torch.save({k: v.detach().cpu() for k, v in best_state.items()},
                   out_dir / 'model.pt')
        (out_dir / 'config.json').write_text(json.dumps(
            {'cfg': cfg, 'dataset': dataset.key, 'seed': seed,
             'data_fraction': data_fraction, 'fold': fold, 'n_folds': n_folds,
             'val_indices': [int(i) for i in i_vl],
             'val_loss': best, 'n_params': n_params}, indent=2),
            encoding='utf-8')
        (out_dir / 'history.json').write_text(json.dumps(history, indent=2),
                                              encoding='utf-8')
        if verbose:
            print(f'  saved -> {out_dir}  (best val {best:.4f})')

    return {'best_val': best, 'history': history, 'n_params': n_params,
            'out_dir': out_dir, 'model': model, 'cfg': cfg, 'fold': fold,
            'n_folds': n_folds}
