"""Fit the Gaussian-process baseline, one sparse variational GP per neuron.

Following Goldin et al. (PNAS 2023): images enter through an arc-cosine kernel
whose covariance is shaped by a learnable Gaussian envelope over pixels, centred
on the neuron's receptive field, with a smoothness parameter setting the finest
spatial scale the response may depend on. Spike counts are modelled with a
Poisson likelihood through an exponential link, and the ELBO is maximised by an
EM-style alternation of variational E-steps, hyper-parameter M-steps and
likelihood-parameter F-steps.

One call fits one fold of the K-fold cross-validation (`cirf.data.kfold_split`):
the GP is trained on the K-1 training folds only, and the held-out fold is
scored out-of-fold by the evaluator. Checkpoints go to ``<out_dir>/cell_XX.pt``;
each one records the indices of the pool images it was trained on, so a fold
(or a subset run of the data-efficiency study) is reconstructible from the
same pool.

Requires gpytorch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from cirf import config as cfg_mod
from cirf import data as dat

from .eigenspace_checkpoint import save_eigenspace_checkpoint
from .eigenspace_model import DirectVGPModel
from .eigenspace_training import train_eigenspace, predict_eigenspace
from .kernels import create_kernel
from .likelihoods import PoissonLikelihood
from .metrics import compute_pearson_correlation


#: Paper Table 3. Identical for both datasets; only the cells and their RF
#: centres change.
DEFAULT_CFG = {
    'mode': 'vargp_direct',
    'M': 300,                     # inducing points
    'n_iterations': 80,
    'n_estep': 50,
    'n_fstep': 10,
    'n_mstep': 20,
    'lr': 0.1,
    'optimizer': 'lbfgs',
    'kernel_type': 'arc_cosine',
    'sigma_0': 1.0,
    'Amp': 1.0,
    'beta': 0.1,                  # RF envelope width
    'rho': 0.1,                   # smoothness
    'rf_init': 'ground_truth',
    'eps_0x': None,               # set per cell from the RF centre
    'eps_0y': None,
    'gradient_mode': 'autograd',
    'use_mask': True,
    # NOTE: these two are recorded in every checkpoint but are NOT applied by
    # this pipeline -- nothing calls kernel.set_center_bounds(), so the RF centre
    # is only clamped to the image, [-1, 1]. Kept so old checkpoints keep loading.
    'bound_rf_center': True,
    'n_sigma_rf_bounds': 3.0,
    'A_init': 0.0001,
    'lambda0_init': -1.0,
    'early_stop': True,
    'patience': 15,
    'min_delta_rel': 0.001,
    'min_iterations': 10,
    'restore_best': True,
    'es_metric': 'elbo',
    'eigval_tol': 0.0001,
    'lambda_var_clamp': 1e-6,
    'jitter': 0.0001,
    'cholesky_max_tries': 3,
    'gpy_lbfgs_max_iter': 20,
    'f_mean_max_threshold': 500,
    'f_mean_mean_threshold': 100,
    'n_val_split': 0,
    'interleave_fstep': True,
    'fix_Amp': True,
    'ip_selection': 'random',
    'pool_sum_tolerance': 0.001,
}


def get_cfg(ds=None, **overrides) -> dict:
    """Config for a dataset; `n_px_side` follows the dataset's GP crop."""
    cfg = dict(DEFAULT_CFG)
    if ds is not None:
        dataset = cfg_mod.resolve(ds)
        cfg['n_px_side'] = dat.gp_crop(dataset)
        cfg.update(dataset.model_config('GP'))
    cfg.setdefault('n_px_side', cfg_mod.GP_CROP)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def train_cell(cell_id, X_pool, r_pool, X_test, r_test, rf_centers, out_dir,
               device, dtype, cfg, pool_indices=None, dataset_key=''):
    """Fit one cell and write its checkpoint; returns (test correlation, seconds).

    ``pool_indices`` selects the training subset out of the pool (all of it by
    default). The integrity tags are always derived from the *full* pool, so a
    checkpoint can be reloaded against the same file regardless of the subset.
    """
    cfg = dict(cfg)
    cfg['cell'] = cell_id
    cfg['dataset'] = dataset_key
    cfg['device'] = str(device)
    cfg['dtype'] = str(dtype).replace('torch.', '')
    cfg['eps_0x'] = float(rf_centers[cell_id, 0])
    cfg['eps_0y'] = float(rf_centers[cell_id, 1])

    if pool_indices is None:
        pool_indices = torch.arange(X_pool.shape[0], dtype=torch.long)
    X_train = X_pool[pool_indices]
    cfg['n_train'] = int(X_train.shape[0])
    X_tilde = X_train[:cfg['M']]

    r_train = r_pool[pool_indices, cell_id].to(dtype=dtype)
    r_test_mean = r_test[:, :, cell_id].mean(dim=0).to(dtype=dtype)

    kernel = create_kernel(cfg, cfg['n_px_side'],
                           cfg['eps_0x'], cfg['eps_0y']).to(device=device, dtype=dtype)
    likelihood = PoissonLikelihood(A_init=cfg['A_init'],
                                   lambda0_init=cfg['lambda0_init']).to(device=device, dtype=dtype)
    model = DirectVGPModel(
        kernel=kernel, likelihood=likelihood,
        X_train=X_train.to(dtype=dtype), X_tilde=X_tilde.to(dtype=dtype),
        eigval_tol=cfg['eigval_tol'], lambda_var_clamp=cfg['lambda_var_clamp'],
    )
    if cfg.get('fix_Amp', False):
        model.kernel.raw_Amp.requires_grad_(False)

    t0 = time.time()
    train_eigenspace(
        model=model, r=r_train,
        n_iterations=cfg['n_iterations'], n_estep=cfg['n_estep'],
        n_fstep=cfg['n_fstep'], n_mstep=cfg['n_mstep'],
        lr_f=cfg['lr'], lr_m=cfg['lr'], print_every=10,
        early_stop=cfg['early_stop'], patience=cfg['patience'],
        min_delta_rel=cfg['min_delta_rel'], min_iterations=cfg['min_iterations'],
        restore_best=cfg['restore_best'], fix_Amp=cfg['fix_Amp'],
        interleave_fstep=cfg['interleave_fstep'],
        f_mean_max_threshold=cfg['f_mean_max_threshold'],
        f_mean_mean_threshold=cfg['f_mean_mean_threshold'],
        es_metric=cfg['es_metric'],
    )
    elapsed = time.time() - t0

    with torch.no_grad():
        f_pred = predict_eigenspace(model, X_test)['f_pred']
        test_r = float(compute_pearson_correlation(r_test_mean, f_pred))

    save_eigenspace_checkpoint(
        model=model, config=cfg, metrics={'test_r': test_r, 'elapsed': elapsed},
        pool_indices=pool_indices, X_pool=X_pool.cpu(),
        checkpoint_path=Path(out_dir) / f'cell_{cell_id:02d}.pt',
    )
    return test_r, elapsed


def train(ds=None, cfg=None, out_dir=None, cells=None, device=None,
          data_fraction=1.0, seed=None, fold=0, n_folds=None, verbose=True):
    """Fit every cell on one fold; returns {'test_r': (N,), 'out_dir': ...}.

    ``fold`` / ``n_folds`` pick the held-out fold of the K-fold cross-validation;
    ``data_fraction`` restricts the training folds the same way as for the
    other models, with the same reproducible subset rule, so the
    data-efficiency study is comparable across architectures.
    """
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    cfg = dict(cfg or get_cfg(dataset))
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    dtype = torch.float32
    n_folds = int(n_folds or cfg_mod.N_FOLDS)
    fold = int(fold)
    out_dir = cfg_mod.ensure(out_dir or cfg_mod.fold_dir('GP', fold, dataset))

    X_pool = dat.gp_pool(dataset, device, dtype)
    r_pool = dat.gp_pool_responses(dataset, device, dtype)
    X_test, r_test = dat.gp_test(dataset, device, dtype)
    rf_centers = dat.gp_rf_centers(dataset)

    i_tr, i_vl = dat.fold_indices(dataset, fold, n_folds, data_fraction,
                                  42 if seed is None else seed)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    pool_indices = torch.as_tensor(i_tr, dtype=torch.long)

    N = r_pool.shape[1]
    cells = list(range(N)) if cells is None else list(cells)
    if verbose:
        print(f'GP on {dataset.label}: {len(cells)} cells, fold {fold}/{n_folds}, '
              f'{len(pool_indices)}/{X_pool.shape[0]} pool images to fit on '
              f'({len(i_vl)} held out), device {device}')

    results = []
    for cell_id in cells:
        if verbose:
            print(f'\n--- cell {cell_id:02d} ---', flush=True)
        test_r, elapsed = train_cell(
            cell_id, X_pool, r_pool, X_test, r_test, rf_centers, out_dir,
            device, dtype, cfg, pool_indices, dataset.key)
        results.append(test_r)
        if verbose:
            print(f'  cell {cell_id:02d}: test r = {test_r:.4f}  ({elapsed:.0f}s)')

    (Path(out_dir) / 'config.json').write_text(json.dumps(
        {'cfg': {k: v for k, v in cfg.items() if k not in ('eps_0x', 'eps_0y')},
         'dataset': dataset.key, 'data_fraction': data_fraction,
         'seed': seed, 'fold': fold, 'n_folds': n_folds,
         'val_indices': [int(i) for i in i_vl], 'n_cells': len(cells)},
        indent=2), encoding='utf-8')
    if verbose:
        print(f'\nmean test r = {np.mean(results):.4f}   -> {out_dir}')
    return {'test_r': np.asarray(results), 'out_dir': out_dir, 'cfg': cfg,
            'fold': fold, 'n_folds': n_folds}
