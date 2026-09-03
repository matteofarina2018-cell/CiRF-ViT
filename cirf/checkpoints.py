"""Reconstructing trained models from a checkpoint directory.

Every model family writes its own layout, so one loader per family:

    transformer / cnn   model.pt + config.json (a CiRF-ViT variant of the
                        ablation records its name and is rebuilt as such)
    GP                  cell_XX.pt (one sparse GP per neuron)
    LN                  cell_XX/seed_K.pt (repeats per neuron)

All four take a directory and a dataset, and return objects the evaluator and
the LSTA code can use directly. A directory holds ONE fold of the K-fold
cross-validation (``results/<dataset>/checkpoints/<model>/fold_K/``); with no
directory given, the loaders take fold ``fold`` (0 by default), and `fold_dirs`
lists every trained fold so a caller can loop over them.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch

from . import config as cfg_mod
from . import data as dat


_CELL_RE = re.compile(r'cell_(\d+)')
_SEED_RE = re.compile(r'seed_(\d+)')


def _numbered(paths, pattern):
    """``{index: path}`` for names like ``cell_07.pt`` or ``seed_3.pt``.

    The index is parsed, never inferred from the sort order of the file names:
    those are zero-padded to two digits, so as soon as a recording has a hundred
    neurons ``cell_100`` sorts *before* ``cell_99`` and every per-neuron
    checkpoint would be attributed to the wrong cell -- silently, since the
    shapes still line up.
    """
    out = {}
    for p in paths:
        m = pattern.fullmatch(p.stem if p.suffix else p.name)
        if m is not None:
            out[int(m.group(1))] = p
    return out


def cell_paths(model_dir, pattern='cell_*.pt') -> list:
    """Per-neuron checkpoint paths, ordered by the neuron they belong to."""
    found = _numbered(Path(model_dir).glob(pattern), _CELL_RE)
    return [found[i] for i in sorted(found)]


def load_run_info(model_dir, name='config.json') -> dict:
    """Everything ``config.json`` records: cfg, dataset, seed, fold, ..."""
    path = Path(model_dir) / name
    if not path.exists():
        raise FileNotFoundError(f'{path} not found (was this model trained?)')
    return json.loads(path.read_text(encoding='utf-8'))


def load_config(model_dir, name='config.json') -> dict:
    """The hyper-parameters a checkpoint was trained with."""
    raw = load_run_info(model_dir, name)
    return raw.get('cfg', raw)


def run_fold(model_dir) -> tuple:
    """(fold, n_folds) a checkpoint was trained on.

    Read from ``config.json``; a checkpoint written before the cross-validation
    existed carries neither and is refused, since nothing says which stimuli it
    was fitted on.
    """
    info = load_run_info(model_dir)
    if 'fold' not in info or 'n_folds' not in info:
        raise ValueError(
            f'{model_dir} records no fold: it predates the K-fold '
            f'cross-validation. Retrain it (scripts/train.py).')
    return int(info['fold']), int(info['n_folds'])


def held_out_indices(model_dir, ds=None):
    """(n_val,) indices into the pool of the stimuli a checkpoint never saw.

    Taken from ``config.json`` when recorded, otherwise rebuilt from the fold
    and the dataset; the two agree by construction, the record is there so a
    checkpoint stays self-describing.
    """
    import numpy as np
    info = load_run_info(model_dir)
    if info.get('val_indices') is not None:
        return np.asarray(info['val_indices'], dtype=np.int64)
    fold, n_folds = run_fold(model_dir)
    return dat.kfold_split(dat.meta(ds)['n_pool'], fold, n_folds)[1]


def fold_dirs(model_type, ds=None, root=None) -> list:
    """[(fold, directory)] of every trained fold, in fold order.

    Looks under ``root`` (default: the model's checkpoint directory) for
    ``fold_K/`` directories holding a complete checkpoint.
    """
    root = Path(root or cfg_mod.checkpoint_dir(model_type, ds))
    found = _numbered([p for p in root.glob('fold_*') if p.is_dir()],
                      re.compile(r'fold_(\d+)'))
    return [(k, found[k]) for k in sorted(found) if exists(model_type, found[k])]


def folds(model_type, ds=None, root=None) -> list:
    """The fold indices that have a trained checkpoint."""
    return [k for k, _ in fold_dirs(model_type, ds, root)]


def load_state_dict(path, device='cpu'):
    """A state dict saved either bare or wrapped, with compile prefixes dropped."""
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    return {k.removeprefix('_orig_mod.'): v for k, v in state.items()}


def load_torch_model(model_type, model_dir=None, ds=None, device='cpu',
                     model_file='model.pt', config_file='config.json', fold=0):
    """Rebuild a transformer or CNN checkpoint; returns (model, cfg)."""
    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    model_dir = Path(model_dir or cfg_mod.fold_dir(model_type, fold, dataset))
    info = load_run_info(model_dir, config_file)
    cfg = info.get('cfg', info)
    cfg.setdefault('IMG_SIZE', dat.img_size(dataset))

    if model_type == 'transformer':
        from models.transformer.variants import build_variant
        model = build_variant(info.get('variant', 'full'), cfg, dataset, device)
    elif model_type == 'cnn':
        from models.cnn.train import build_model
        model = build_model(cfg, dataset, device)
    else:
        raise ValueError(f'{model_type!r} is not a torch checkpoint')
    model.load_state_dict(load_state_dict(model_dir / model_file, device))
    model.eval()
    return model, cfg


def load_gp_models(model_dir=None, ds=None, device='cpu', n_cells=None, fold=0):
    """Rebuild every per-cell GP; returns (models, predict_fn).

    The checkpoints store the pool they were trained on by index plus an
    integrity tag, so the same pool has to be rebuilt from the dataset here.
    """
    from models.gp.eigenspace_checkpoint import load_eigenspace_checkpoint
    from models.gp.eigenspace_training import predict_eigenspace

    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    model_dir = Path(model_dir or cfg_mod.fold_dir('GP', fold, dataset))
    X_pool = dat.gp_pool(dataset, device)
    tol = dat.gp_pool_tolerance(X_pool)

    files = cell_paths(model_dir)
    if n_cells is not None:
        files = files[:n_cells]
    if not files:
        raise FileNotFoundError(f'no cell_*.pt in {model_dir}')
    if n_cells is None:
        _require_all_cells('GP', files, dataset, model_dir)

    models = [load_eigenspace_checkpoint(checkpoint_path=str(f), X_pool=X_pool,
                                         pool_sum_tolerance=tol, device=device)['model']
              for f in files]
    return models, predict_eigenspace


def load_ln_models(model_dir=None, ds=None, device='cpu', fold=0):
    """Rebuild the per-cell LN repeats of one fold; returns (cells, cfg).

    `cells[c]` is the list of that neuron's fitted repeats, in seed order.
    """
    from models.ln.model import LNModel

    dataset = cfg_mod.resolve(ds or cfg_mod.active())
    model_dir = Path(model_dir or cfg_mod.fold_dir('LN', fold, dataset))
    cfg = load_config(model_dir)
    crop = int(cfg['CROP_SIZE'])

    dirs = [d for d in cell_paths(model_dir, 'cell_*') if d.is_dir()]
    cells = []
    for cell_dir in dirs:
        by_seed = _numbered(cell_dir.glob('seed_*.pt'), _SEED_RE)
        seeds = []
        for k in sorted(by_seed):
            ck = torch.load(by_seed[k], map_location=device, weights_only=False)
            m = LNModel(crop).to(device)
            m.load_state_dict(ck['state_dict'])
            m.eval()
            seeds.append(m)
        cells.append(seeds)
    if not cells:
        raise FileNotFoundError(f'no cell_*/seed_*.pt in {model_dir}')
    _require_all_cells('LN', dirs, dataset, model_dir)

    # Every neuron must carry the same number of repeats. `cirf.evaluate` sizes
    # its prediction array from the largest count and averages over that axis,
    # so a neuron left with fewer -- an interrupted fit -- would be averaged
    # against zeros and silently scaled down.
    counts = {len(seeds) for seeds in cells}
    if len(counts) != 1:
        short = [(int(d.name.split('_')[1]), len(seeds))
                 for d, seeds in zip(dirs, cells) if len(seeds) != max(counts)]
        raise ValueError(
            f'{model_dir}: the LN repeats are uneven -- {max(counts)} for most '
            f'neurons but {short[:6]} for others (neuron, repeats). Averaging '
            f'over them would scale those neurons down. Finish or delete the '
            f'run.')
    return cells, cfg


def _require_all_cells(model_type, paths, dataset, model_dir):
    """Refuse a per-neuron checkpoint directory that does not cover every neuron.

    The GP and the LN write one file per neuron and the loaders return them in
    neuron order, so a half-finished run (an interrupted fit, or one restricted
    with ``--cells``) would otherwise be scored as if it were the whole
    population: entry `i` of the checkpoint list would be compared against
    neuron `i` of the recording, and the metrics would average over the subset
    without saying so. Checking the indices, and not merely how many there are,
    also catches a directory holding the right *number* of the wrong cells.
    """
    expected = dat.n_cells(dataset)
    found = sorted(int(_CELL_RE.fullmatch(p.stem if p.suffix else p.name).group(1))
                   for p in paths)
    if found != list(range(expected)):
        missing = sorted(set(range(expected)) - set(found))
        raise ValueError(
            f'{model_dir} holds {len(found)} of the {expected} neurons of '
            f'{dataset.key}'
            + (f' (missing {missing[:8]}{"..." if len(missing) > 8 else ""})'
               if missing else '')
            + f': a partial {model_type} run cannot be scored, its cells would '
            f'be matched to the wrong neurons. Finish the run '
            f'(scripts/train.py --model {model_type} --dataset {dataset.key}) '
            f'or delete the directory.')


def load_ln_folds(ds=None, device='cpu', root=None) -> list:
    """[(fold, cells, cfg)] for every trained LN fold -- the figures' loop."""
    return [(k, *load_ln_models(d, ds, device))
            for k, d in fold_dirs('LN', ds, root)]


def mirror(src, dst) -> Path:
    """Copy a run's checkpoint files from ``src`` to ``dst`` (its twin location).

    The canonical run of a fold and the data-efficiency run at 100 % of the
    data with the canonical seed are one and the same training, kept at two
    paths so that either script finds it and neither trains it again. Every
    file of the run is copied except its ``eval/`` directory, which is
    produced per location; files already in ``dst`` are overwritten, nothing
    else there is touched.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        return dst
    for p in src.rglob('*'):
        rel = p.relative_to(src)
        if rel.parts[0] == 'eval':
            continue
        if p.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
        else:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)
    return dst


def exists(model_type, model_dir=None, ds=None) -> bool:
    """Whether a trained checkpoint is on disk.

    With ``model_dir`` given, that directory (one fold) is checked; without,
    whether at least one fold of the model's cross-validation has been trained.
    """
    if model_dir is None:
        root = cfg_mod.checkpoint_dir(model_type, ds)
        return any(exists(model_type, p) for p in root.glob('fold_*') if p.is_dir())
    d = Path(model_dir)
    if model_type == 'GP':
        return any(d.glob('cell_*.pt'))
    if model_type == 'LN':
        return any(d.glob('cell_*/seed_*.pt'))
    return (d / 'model.pt').exists()
