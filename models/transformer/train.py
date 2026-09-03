"""Build and train CiRF-ViT on any dataset.

    from models.transformer import build_model, train
    train('mydata.npz')                       # one run, defaults, saved
    model = build_model(get_cfg(), 'mydata.npz')

Training is plain Adam on the Poisson NLL with early stopping on the validation
loss and best-weight restore -- no schedule, no augmentation, nothing dataset
specific. One call fits one fold of the K-fold cross-validation: the held-out
fold is the validation set, the other K-1 folds the training set
(`cirf.data.kfold_split`); scripts/train.py loops over the folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from cirf import config as cfg_mod
from cirf import data as dat

from .config import get_cfg
from .losses import poisson_loss
from .model import RetinalModel


def build_model(cfg: dict, ds=None, device='cpu') -> RetinalModel:
    """Instantiate the model for a dataset, reading geometry from the data.

    The RF centres initialise the crop and the image size is recorded in the
    config, so a checkpoint can be rebuilt without the data.
    """
    cx, cy = dat.centers(ds)
    n_cells = len(cx)
    cfg = dict(cfg)
    cfg.setdefault('IMG_SIZE', dat.img_size(ds))

    from .config import resolve_radii
    radii = np.asarray(resolve_radii(cfg, n_cells), dtype=np.float32)
    cfg['NEURON_RADII'] = [int(r) for r in radii]

    return RetinalModel(cfg, cx, cy, radii).to(device)


def save_checkpoint(model, cfg, out_dir, history=None, **info) -> Path:
    """Write ``model.pt``, ``config.json`` and ``history.json`` to ``out_dir``.

    The layout every run of this family shares -- the canonical folds, the
    data-efficiency runs, the ablation variants, the no-crop comparison -- so
    that `cirf.checkpoints.load_torch_model` rebuilds any of them. ``info``
    (dataset, seed, fold, variant, ...) is recorded next to the config.
    """
    out_dir = cfg_mod.ensure(out_dir)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
               out_dir / 'model.pt')
    (out_dir / 'config.json').write_text(json.dumps({'cfg': cfg, **info}, indent=2),
                                         encoding='utf-8')
    if history is not None:
        (out_dir / 'history.json').write_text(json.dumps(history, indent=2),
                                              encoding='utf-8')
    return out_dir


def fit(model, train_loader, val_loader, cfg, device='cpu', early_stop=True,
        epochs=None, verbose=True):
    """Train any model of this family; returns (best val, history, epochs run).

    Adam on the Poisson NLL, early stopping on the validation loss with
    best-weight restore. `early_stop=False` runs the full `epochs` regardless,
    which is what the crop / no-crop comparison needs to show both curves over
    the same range. The model is left holding its best weights.

    With `cfg['COMPILE']` the forward and backward passes run through
    `torch.compile` on a GPU; the compiled wrapper shares the parameters of
    `model`, which is what is returned, saved and evaluated.
    """
    epochs = int(epochs or cfg['MAX_EPOCHS'])
    optimizer = Adam(model.parameters(), lr=cfg['LEARNING_RATE'],
                     weight_decay=cfg['WEIGHT_DECAY'])
    runner = model
    if cfg.get('COMPILE') and torch.device(device).type == 'cuda':
        runner = torch.compile(model)
    best_val, patience, best_state, last_epoch = float('inf'), 0, None, 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        runner.train()
        tr_loss = 0.0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            resp = batch['response'].to(device)
            optimizer.zero_grad()
            loss = poisson_loss(runner(imgs), resp)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        runner.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                resp = batch['response'].to(device)
                vl_loss += poisson_loss(runner(imgs), resp).item()
        vl_loss /= len(val_loader)

        history['train_loss'].append(float(tr_loss))
        history['val_loss'].append(float(vl_loss))
        if verbose:
            print(f'  epoch {epoch:4d}  train {tr_loss:.4f}  val {vl_loss:.4f}',
                  flush=True)

        if vl_loss < best_val:
            best_val, patience = vl_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            history['best_epoch'] = epoch
        else:
            patience += 1
            if early_stop and patience >= cfg['EARLY_STOP']:
                if verbose:
                    print(f'  early stop at epoch {epoch} (best val {best_val:.4f})')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, history, last_epoch


def train(ds=None, cfg=None, out_dir=None, seed=None, data_fraction=1.0,
          device=None, max_epochs=None, fold=0, n_folds=None, save=True,
          verbose=True):
    """Train one fold; returns {'best_val', 'history', 'n_params', 'out_dir'}.

    ds            dataset name, alias or path to an .npz
    cfg           config dict; None -> `get_cfg(ds)`
    out_dir       where model.pt / config.json / history.json go;
                  None -> results/<dataset>/checkpoints/transformer/fold_<fold>/
    seed          torch/numpy seed; also selects the training subset when
                  `data_fraction < 1`
    data_fraction share of the training folds to use
    fold          which fold of the K-fold cross-validation is held out
    n_folds       K (default cfg.N_FOLDS)
    """
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    cfg = dict(cfg or get_cfg(dataset))
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    n_folds = int(n_folds or cfg_mod.N_FOLDS)
    fold = int(fold)
    out_dir = Path(out_dir or cfg_mod.fold_dir('transformer', fold, dataset))
    if max_epochs:
        cfg['MAX_EPOCHS'] = int(max_epochs)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    train_loader, val_loader, _ = dat.get_dataloaders(
        dataset, batch_size=cfg['BATCH_SIZE'], data_fraction=data_fraction,
        random_seed=42 if seed is None else seed, fold=fold, n_folds=n_folds)
    _, val_indices = dat.fold_indices(dataset, fold, n_folds)

    model = build_model(cfg, dataset, device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f'CiRF-ViT on {dataset.label}: {model.n_cells} neurons, '
              f'{n_params:,} parameters, fold {fold}/{n_folds} '
              f'({len(train_loader.dataset)} train / {len(val_indices)} held out), '
              f'device {device}')

    best_val, history, _ = fit(model, train_loader, val_loader, cfg, device,
                               verbose=verbose)

    if save:
        save_checkpoint(model, cfg, out_dir, history, dataset=dataset.key,
                        seed=seed, data_fraction=data_fraction, fold=fold,
                        n_folds=n_folds, val_indices=[int(i) for i in val_indices],
                        val_loss=best_val, n_params=n_params)
        if verbose:
            print(f'  saved -> {out_dir}  (best val {best_val:.4f})')

    return {'best_val': best_val, 'history': history, 'n_params': n_params,
            'out_dir': out_dir, 'model': model, 'cfg': cfg, 'fold': fold,
            'n_folds': n_folds}
