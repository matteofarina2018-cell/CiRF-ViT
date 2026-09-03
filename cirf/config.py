"""Paths, dataset registry and shared constants.

Everything the pipeline needs to find on disk is resolved here, so no script
hard-codes a location. Two roots can be redirected with environment variables:

    CIRF_DATA_DIR      where dataset .npz files are looked up   (default ./data)
    CIRF_RESULTS_DIR   where everything produced is written     (default ./results)

Neither directory ships with the code: recordings and trained checkpoints are
not part of this repository.

Datasets
--------
A dataset is a single .npz following the schema in ``docs/DATA_FORMAT.md``. Two
are pre-registered because the paper uses them (``mice`` / ``salamander``), but
they are not special: any file with the same layout works, either by passing
``--data my_recording.npz`` to any script or by registering it once,

    from cirf import config as cfg
    cfg.register('mydata', '/path/to/mydata.npz', label='my retina')

Nothing here fixes the number of neurons, the image size or the number of
stimuli -- those are read from the file (see ``cirf.data``).

The active dataset is module state, set once per run by ``set_dataset()`` and
read through ``active()`` by every module that needs it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get('CIRF_DATA_DIR', ROOT / 'data'))
RESULTS_DIR = Path(os.environ.get('CIRF_RESULTS_DIR', ROOT / 'results'))


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    """A recording the pipeline can run on.

    key        short name; every output file and directory is keyed by it
    npz_name   file name (looked up under DATA_DIR) or an absolute path
    label      human-readable name for figure titles and log lines
    config     per-model hyper-parameter overrides, ``{'transformer': {...}}``;
               the tuned settings of the two paper datasets live here.
    """

    key: str
    npz_name: str
    label: str = ''
    config: dict = field(default_factory=dict)

    @property
    def npz(self) -> Path:
        p = Path(self.npz_name)
        return p if p.is_absolute() else DATA_DIR / p

    @property
    def root(self) -> Path:
        return RESULTS_DIR / self.key

    def model_config(self, model_type: str) -> dict:
        """Tuned overrides for one model on this dataset (possibly empty)."""
        return dict(self.config.get(model_type, {}))


DATASETS: dict = {}
ALIASES: dict = {}


def register(key, npz_name, label='', config=None,
             aliases=(), overwrite=False) -> Dataset:
    """Add a dataset to the registry and return it."""
    if key in DATASETS and not overwrite:
        raise ValueError(f'dataset {key!r} is already registered')
    ds = Dataset(key, str(npz_name), label or key, dict(config or {}))
    DATASETS[key] = ds
    ALIASES[key.lower()] = key
    for a in aliases:
        ALIASES[str(a).lower()] = key
    return ds


def register_path(path, key=None, **kwargs) -> Dataset:
    """Register an .npz by path (its stem becomes the key unless given).

    This is what ``--data`` does on every script, so an unseen recording can be
    trained and benchmarked without touching the code.
    """
    path = Path(path).expanduser().resolve()
    key = key or path.stem
    kwargs.setdefault('overwrite', True)     # re-registering a path is normal
    return register(key, path, **kwargs)


# The mouse dataset of Goldin et al. (PNAS 2023): 41 ganglion cells.
register('mice', 'Mice_dataset.npz', label='mouse',
         aliases=('mouse',))

# Two salamander preparations pooled: 49 cells.
# Its tuned settings differ from the defaults on two points, both from the same
# Bayesian search that produced the mouse ones.
register('salamander', 'Salamander_dataset.npz', label='salamander',
         aliases=('axolotl',),
         config={
             'transformer': {'CNN_KERNEL': 11},
             'cnn': {'LAMBDA_FEATURE': 2e-4},
         })

DATASET_CHOICES = ('mice', 'mouse', 'salamander')

_active: Dataset = DATASETS['mice']


def resolve(name) -> Dataset:
    """Dataset from a name, an alias, a path to an .npz, or a Dataset."""
    if isinstance(name, Dataset):
        return name
    text = str(name)
    key = ALIASES.get(text.lower())
    if key is not None:
        return DATASETS[key]
    if text.lower().endswith('.npz'):
        return register_path(text)
    raise ValueError(f'unknown dataset {name!r}: register it, pass a path to '
                     f'an .npz, or choose from {sorted(DATASETS)}')


def set_dataset(name) -> Dataset:
    """Make `name` the active dataset for this process and return it."""
    global _active
    _active = resolve(name)
    return _active


def active() -> Dataset:
    return _active


def expand(name) -> list:
    """`'both'`/`'all'` -> every registered dataset, otherwise just this one."""
    if str(name).lower() in ('both', 'all'):
        return list(DATASETS.values())
    return [resolve(name)]


# -----------------------------------------------------------------------------
# Output layout (everything below RESULTS_DIR)
#
#   results/<dataset>/checkpoints/<model>/fold_K/       one fold of the K-fold CV
#   results/<dataset>/checkpoints/<model>/fold_K/eval/  that fold's eval_data.npz
#   results/<dataset>/checkpoints/<model>/eval/         eval_data.npz aggregated
#                                                       over the folds
#   results/<dataset>/fraction_study/<model>/frac_XXX/fold_K/seed_S/
#       the run at frac_100 with TRAIN_SEED is the same training as the
#       canonical fold_K above; whichever script trains it writes both copies
#   results/<dataset>/experiments/                 ablation.json, history_nocrop.json
#   results/<dataset>/experiments/ablation/<variant>/fold_K/seed_S/
#                                                  every ablation run's checkpoint
#   results/<dataset>/experiments/crop_vs_nocrop/fold_K/   the no-crop run's
#   results/<dataset>/cache/                       derived arrays reused by figures
#   results/figures/{paper,extra}/                 the figures themselves
# -----------------------------------------------------------------------------

MODEL_TYPES = ('GP', 'cnn', 'transformer', 'LN')


def checkpoint_dir(model_type: str, ds=None) -> Path:
    """Root of a model's K-fold run: one ``fold_K/`` per fold, ``eval/`` on top."""
    return resolve(ds or active()).root / 'checkpoints' / model_type


def fold_dir(model_type: str, fold: int, ds=None) -> Path:
    """The checkpoint directory of one fold of the cross-validation."""
    return checkpoint_dir(model_type, ds) / f'fold_{int(fold)}'


def eval_path(model_type: str, ds=None) -> Path:
    """The evaluation aggregated over every fold (what the figures read)."""
    return checkpoint_dir(model_type, ds) / 'eval' / 'eval_data.npz'


def fold_eval_path(model_type: str, fold: int, ds=None) -> Path:
    return fold_dir(model_type, fold, ds) / 'eval' / 'eval_data.npz'


def fraction_root(model_type: str, ds=None) -> Path:
    return resolve(ds or active()).root / 'fraction_study' / model_type


def fraction_dir(model_type: str, frac: float, fold: int, seed: int = 0,
                 ds=None) -> Path:
    """One run of the data-efficiency sweep: a fraction, a fold and a seed."""
    return (fraction_root(model_type, ds)
            / f'frac_{int(round(frac * 100)):03d}' / f'fold_{int(fold)}'
            / f'seed_{int(seed)}')


def experiment_dir(ds=None) -> Path:
    return resolve(ds or active()).root / 'experiments'


def ablation_dir(variant: str, fold: int, seed: int = 0, ds=None) -> Path:
    """Checkpoint of one ablation run (scripts/ablation.py)."""
    return (experiment_dir(ds) / 'ablation' / variant / f'fold_{int(fold)}'
            / f'seed_{int(seed)}')


def nocrop_dir(fold: int, ds=None) -> Path:
    """Checkpoint of the no-crop training-dynamics run (scripts/crop_vs_nocrop.py)."""
    return experiment_dir(ds) / 'crop_vs_nocrop' / f'fold_{int(fold)}'


def cache_dir(ds=None) -> Path:
    return resolve(ds or active()).root / 'cache'


def figure_dir(kind: str = 'paper') -> Path:
    return RESULTS_DIR / 'figures' / kind


def ensure(path) -> Path:
    """mkdir -p for a directory, returning it."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


# -----------------------------------------------------------------------------
# Study-wide constants
#
# Image size and neuron count are NOT here: they are properties of the data and
# are read from the file by `cirf.data`.
# -----------------------------------------------------------------------------

GP_CROP = 64                # side of the centred crop the GP model sees

# A recording ships ONE pool of stimuli, not a train/validation split. Every
# model is fitted by classic K-fold cross-validation over that pool: the pool is
# partitioned once, by a seeded permutation, into N_FOLDS folds of equal size;
# fold k is held out while the model is fitted on the other K-1 (the held-out
# fold is what early stopping watches and what the out-of-fold score is
# computed on), and the K fold models are each scored on the repeated test
# stimuli. The partition is identical for every model, every seed and every
# run, so a checkpoint is reproducible from the file alone and the folds are
# comparable across architectures.
N_FOLDS = 5                 # K of the K-fold cross-validation
KFOLD_SEED = 42             # fixed: the partition must not vary between runs
FOLDS = list(range(N_FOLDS))

TRAIN_SEED = 0              # seed of the canonical run of every fold. The
                            # data-efficiency sweep at 100 % of the data with
                            # this seed is that same training, so the two are
                            # written to both locations and never trained twice

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FRAC_SEEDS = [TRAIN_SEED]       # training seeds per (fraction, fold); a sweep
                                # run is one fold x one seed, and every one of
                                # them is shown -- no run is ever dropped

STIM_DURATION_S = 0.3       # seconds of image presentation; spike counts -> Hz


# -----------------------------------------------------------------------------
# Naming and colours
# -----------------------------------------------------------------------------

MODEL_LABELS = {'GP': 'GP', 'cnn': 'CNN', 'transformer': 'CiRF-ViT', 'LN': 'LN'}
MODEL_ABBR = {'GP': 'GP', 'cnn': 'CNN', 'transformer': 'CiRF', 'LN': 'LN'}

#: palette of the paper figures
PAPER_COLORS = {'transformer': '#C83C37', 'cnn': '#0F5450',
                'GP': '#772C54', 'LN': '#FFC000'}

#: Okabe-Ito, colourblind-safe; used by the supplementary figures
OKABE_ITO = {
    'orange': '#E69F00', 'sky_blue': '#56B4E9', 'green': '#009E73',
    'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00',
    'purple': '#CC79A7', 'black': '#000000',
}

MODEL_COLORS = {'GP': OKABE_ITO['green'], 'cnn': OKABE_ITO['blue'],
                'transformer': OKABE_ITO['vermillion'], 'LN': OKABE_ITO['purple']}
MODEL_MARKERS = {'GP': 'o', 'cnn': 's', 'transformer': '^', 'LN': 'D'}
