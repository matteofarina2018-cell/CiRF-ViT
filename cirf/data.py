"""Dataset loading. One .npz per recording, self-contained.

Everything the models and the benchmarks need lives in a single file, so there
is no second data source anywhere in the pipeline. The full schema, including
which fields are optional and what each analysis needs, is in
``docs/DATA_FORMAT.md``; the short version (N neurons, P pixels per side):

    images_pool               (n, P, P, 1)   the fitting stimuli, any P
    responses_pool            (n, N)         spike counts per stimulus
    images_test               (n_test, P, P, 1)    held-out stimuli
    responses_test            (n_rep, n_test, N)   repeated presentations
    centers_x / centers_y     (N,)           RF centres, in image pixels
    lsta                      (N, n_img, S, S)  optional, for the LSTA benchmark
    lsta_base_images          (n_img, P, P)     the stimuli it was measured on
    ellipses                  (N, 2, 360)    optional, RF outlines
    ellipse_px                scalar         the frame `ellipses` is in, if not P

That is the whole schema: a file holds what was measured and nothing else.
Names -- which cluster a neuron was, which preparation, what the stimuli were
called -- are in `STIMULUS_IDS` / `CELL_INDICES` / `EXP_ID` below. Sizes are
shapes, read by `meta()`.

A recording carries ONE pool of fitting stimuli. Every model is fitted by
K-fold cross-validation over it: `kfold_split` partitions the pool once, with a
fixed seed, into `cfg.N_FOLDS` folds; fold k is held out (early stopping and
the out-of-fold score) while the model is fitted on the other K-1. The dataset
file never fixes an arbitrary partition, and every model sees the same folds.

Nothing is hard-coded to a neuron count, an image size or a number of repeats:
those are read from the file by `meta()` and flow through the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from . import config as cfg

REQUIRED = ('images_pool', 'responses_pool', 'images_test', 'responses_test',
            'centers_x', 'centers_y')
OPTIONAL = ('lsta', 'lsta_base_images', 'ellipses', 'ellipse_px')

_FILES: dict = {}
_META: dict = {}


# Bookkeeping about the paper's two recordings: not measurements, so not in the
# dataset files. Keyed by dataset key; an unseen .npz needs none of it.

#: What the experiment called the LSTA stimuli, in `lsta` image order. Labels,
#: not positions: the salamander's 30 is not in that recording's test set.
STIMULUS_IDS = {
    'mice': (16, 0, 1, 19, 8, 12, 11, 24),
    'salamander': (30, 0, 1, 21),
}

#: Identifier of each neuron in the source recording, in file order. Salamander:
#: spike-sorting cluster numbers. Mouse: inherited from the preparation, in a
#: numbering that is not the lab's sorting clusters. Reference only.
CELL_INDICES = {
    'mice': (41, 54, 118, 141, 144, 17, 28, 45, 50, 51, 56, 57, 60, 62, 63, 75,
             81, 82, 86, 99, 104, 110, 113, 115, 122, 124, 133, 134, 142, 147,
             158, 159, 168, 170, 172, 2, 3, 5, 10, 12, 13),
    'salamander': (815, 816, 817, 821, 823, 842, 850, 857, 865, 868, 869, 871,
                   883, 886, 887, 888, 889, 892, 903, 905, 907, 913, 914, 915,
                   916, 917, 109, 197, 999, 1000, 1009, 1014, 1018, 1029, 1042,
                   1046, 1047, 1051, 1055, 1154, 1165, 1167, 1178, 1181, 1182,
                   1183, 1184, 1185, 1186),
}

#: Preparation each neuron came from, for recordings that pool more than one.
EXP_ID = {
    'salamander': (101,) * 26 + (102,) * 23,
}


def stimulus_ids(ds=None, n_img=None):
    """LSTA stimulus labels, in `lsta` image order; 0..n_img-1 if unknown."""
    d = cfg.resolve(ds or cfg.active())
    ids = STIMULUS_IDS.get(d.key)
    if ids is None:
        if n_img is None:
            n_img = meta(d)['n_lsta_images']
        return np.arange(int(n_img))
    return np.asarray(ids, dtype=np.int64)


def load(ds=None):
    """The raw NpzFile for a dataset (memoised per path)."""
    d = cfg.resolve(ds or cfg.active())
    path = d.npz
    if path not in _FILES:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Datasets are not shipped with the code: put "
                f"one in {cfg.DATA_DIR}, pass --data <file.npz>, or point "
                f"CIRF_DATA_DIR at your own directory.")
        _FILES[path] = np.load(path, allow_pickle=True)
    return _FILES[path]


def meta(ds=None) -> dict:
    """Shapes and capabilities read off the file (memoised).

    Keys: n_cells, img_size, n_pool, n_test, n_rep, ellipse_px,
    has_lsta, has_ellipses, n_lsta_images, lsta_px.
    """
    d = cfg.resolve(ds or cfg.active())
    if d.npz in _META:
        return _META[d.npz]

    f = load(d)
    missing = [k for k in REQUIRED if k not in f.files]
    if missing:
        raise KeyError(f'{d.npz.name} is missing required arrays: {missing}. '
                       f'See docs/DATA_FORMAT.md.')

    # Both, or the benchmark cannot run: the maps are meaningless without the
    # stimuli they were measured around. Reporting `has_lsta` only when the file
    # carries both means every downstream script skips the comparison cleanly
    # instead of failing on a missing key; `validate()` says which part is
    # missing.
    has_lsta = 'lsta' in f.files and 'lsta_base_images' in f.files
    has_ell = 'ellipses' in f.files
    img_size = int(f['images_pool'].shape[1])
    ellipse_px = int(f['ellipse_px']) if 'ellipse_px' in f.files else img_size

    info = dict(
        n_cells=int(f['responses_pool'].shape[1]),
        img_size=img_size,
        n_pool=int(f['images_pool'].shape[0]),
        n_test=int(f['images_test'].shape[0]),
        n_rep=int(f['responses_test'].shape[0]),
        ellipse_px=int(ellipse_px),
        has_lsta=has_lsta, has_ellipses=has_ell,
        n_lsta_images=int(f['lsta'].shape[1]) if has_lsta else 0,
        lsta_px=int(f['lsta'].shape[-1]) if has_lsta else 0,
    )
    _META[d.npz] = info
    return info


def n_cells(ds=None) -> int:
    return meta(ds)['n_cells']


def img_size(ds=None) -> int:
    return meta(ds)['img_size']


def ellipse_frame(ds=None) -> int:
    """Pixel frame the RF outlines are stored in (not the LSTA resolution)."""
    return meta(ds)['ellipse_px']


def describe(ds=None) -> str:
    """One-paragraph summary, and which benchmarks the file supports."""
    d = cfg.resolve(ds or cfg.active())
    m = meta(d)
    lines = [
        f'{d.label}  [{d.key}]  {d.npz}',
        f'  {m["n_cells"]} neurons, {m["img_size"]}x{m["img_size"]} stimuli',
        f'  {m["n_pool"]} pool / {m["n_test"]} test images'
        f' ({m["n_rep"]} repeats)',
        f'  LSTA benchmark    : '
        + (f'yes ({m["n_lsta_images"]} stimuli at {m["lsta_px"]} px)'
           if m['has_lsta'] and m['has_ellipses'] else 'no'),
    ]
    if m['has_ellipses'] and m['ellipse_px'] != m['img_size']:
        lines.append(f'  RF outline frame  : {m["ellipse_px"]} px')
    return '\n'.join(lines)


def validate(path) -> dict:
    """Check a candidate .npz against the schema; returns its `meta()`.

    Raises KeyError/ValueError with a specific message when something is off,
    so a user preparing their own recording gets told what to fix.
    """
    ds = cfg.register_path(path)
    m = meta(ds)
    f = load(ds)
    N = m['n_cells']
    for split in ('pool', 'test'):
        img = f[f'images_{split}']
        if img.ndim != 4 or img.shape[1] != img.shape[2]:
            raise ValueError(f'images_{split} must be (n, P, P, 1), got {img.shape}')
        if img.shape[1] != m['img_size']:
            raise ValueError('the pool and the test set must share an image size')
    r = f['responses_pool']
    if r.shape != (f['images_pool'].shape[0], N):
        raise ValueError(f'responses_pool must be (n_pool, {N}), got {r.shape}')
    rt = f['responses_test']
    if rt.ndim != 3 or rt.shape[1:] != (m['n_test'], N):
        raise ValueError(f'responses_test must be (n_rep, n_test, {N}), got {rt.shape}')
    if rt.shape[0] < 2:
        raise ValueError('responses_test needs at least 2 repeats: the metrics '
                         'normalise by the split-half reliability of each neuron')
    for k in ('centers_x', 'centers_y'):
        if f[k].shape != (N,):
            raise ValueError(f'{k} must be ({N},), got {f[k].shape}')
    # `has_lsta` is False unless BOTH arrays are there, so say which is missing
    # rather than silently reporting the benchmark as unavailable.
    if 'lsta' in f.files and 'lsta_base_images' not in f.files:
        raise ValueError('lsta needs lsta_base_images: the stimuli the maps '
                         'were measured around are part of the measurement, so '
                         'they travel with them')
    if 'lsta_base_images' in f.files and 'lsta' not in f.files:
        raise ValueError('lsta_base_images without lsta: the stimuli are there '
                         'but the maps measured around them are not')
    if m['has_lsta']:
        if f['lsta'].ndim != 4 or f['lsta'].shape[0] != N:
            raise ValueError(f'lsta must be ({N}, n_img, S, S), got {f["lsta"].shape}')
        if not m['has_ellipses']:
            raise ValueError('lsta needs ellipses: the comparison is scored '
                             'inside each RF outline')
        if f['ellipses'].shape[:2] != (N, 2):
            raise ValueError(f'ellipses must be ({N}, 2, n_points), '
                             f'got {f["ellipses"].shape}')
        base = f['lsta_base_images']
        if base.shape != (f['lsta'].shape[1], m['img_size'], m['img_size']):
            raise ValueError(f'lsta_base_images must be (n_img, P, P), '
                             f'got {base.shape}')
    # The outlines are in image pixels, like the centres, so they must sit on
    # them. Catches a file whose outlines were fitted in a smaller frame and
    # left unscaled, which misplaces every receptive field at once.
    if m['has_ellipses']:
        e = np.asarray(f['ellipses'], dtype=np.float64)
        off = float(np.median(np.hypot(e[:, 0].mean(1) - f['centers_x'],
                                       e[:, 1].mean(1) - f['centers_y'])))
        if off > 0.1 * m['img_size']:
            raise ValueError(
                f'the RF outlines sit {off:.1f} px away from centers_x/'
                f'centers_y; both must be in image pixels')
    return m


def build_npz(path, images_pool, responses_pool, images_test, responses_test,
              centers_x, centers_y, lsta=None, lsta_base_images=None,
              ellipses=None, ellipse_px=None, **extra):
    """Assemble a dataset file from arrays, then validate it.

    Images may be given as (n, P, P) or (n, P, P, 1); responses as spike counts
    per stimulus. `images_pool` is every stimulus the models may be fitted on --
    there is no train/validation split to supply, `kfold_split` makes the folds.
    `responses_test` is (n_rep, n_test, N): repeated presentations of the
    held-out stimuli, which every metric needs in order to normalise by how
    reproducible each neuron is.

    The optional arrays unlock the LSTA benchmark (`lsta` + `lsta_base_images`
    + `ellipses`), so a minimal recording -- stimuli, responses, RF centres --
    is enough to start. There is nothing else to supply: names for the neurons
    or the stimuli are bookkeeping about one recording, not part of the schema
    (see the module docstring).

        from cirf.data import build_npz
        build_npz('data/my_recording.npz', Xpool, Ypool, Xte, Yte, cx, cy)
    """
    def as_nhwc(a):
        a = np.asarray(a, dtype=np.float32)
        return a[..., None] if a.ndim == 3 else a

    payload = {
        'images_pool': as_nhwc(images_pool),
        'images_test': as_nhwc(images_test),
        'responses_pool': np.asarray(responses_pool, dtype=np.float32),
        'responses_test': np.asarray(responses_test, dtype=np.float32),
        'centers_x': np.asarray(centers_x, dtype=np.float32),
        'centers_y': np.asarray(centers_y, dtype=np.float32),
    }
    optional = {'lsta': lsta, 'ellipses': ellipses,
                'lsta_base_images': lsta_base_images}
    for key, value in optional.items():
        if value is not None:
            payload[key] = np.asarray(value, dtype=np.float32)
    if ellipse_px is not None:
        payload['ellipse_px'] = np.int32(ellipse_px)
    payload.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    _FILES.pop(path, None)
    _META.pop(path, None)
    info = validate(path)
    print(f'wrote {path}: {info["n_cells"]} neurons, '
          f'{info["img_size"]}x{info["img_size"]} stimuli')
    return path


# -----------------------------------------------------------------------------
# Views used all over the pipeline
# -----------------------------------------------------------------------------

def images(split='pool', ds=None):
    """(n, P, P) array for split in {'pool', 'test'}."""
    return load(ds)[f'images_{split}'].squeeze(-1)


def responses(split='pool', ds=None):
    return load(ds)[f'responses_{split}']


def test_repeats(ds=None):
    """(n_test, n_rep, N) -- the repeated test responses, image-major."""
    return load(ds)['responses_test'].transpose(1, 0, 2)


def centers(ds=None):
    d = load(ds)
    return d['centers_x'], d['centers_y']


def lsta_reference(ds=None) -> dict:
    """Experimental LSTA, RF outlines and the labels of the stimuli involved.

    The labels come from `STIMULUS_IDS`, not from the file: they name the
    stimuli in the experiment's own terms, which is bookkeeping about one
    recording rather than something measured in it.
    """
    d = load(ds)
    if not meta(ds)['has_lsta']:
        raise KeyError('this dataset carries no experimental LSTA')
    idx = stimulus_ids(ds, n_img=d['lsta'].shape[1])
    return {'lsta': d['lsta'], 'ellipses': d['ellipses'], 'image_indices': idx}


def lsta_base_images(ds=None):
    """(n_img, P, P) -- the stimuli the experimental LSTA was measured on.

    Always a dedicated array: the maps are a property of the stimulus they were
    measured around, so that stimulus travels with them rather than being looked
    up in a split that may not contain it.
    """
    return load(ds)['lsta_base_images'].astype(np.float32)


# -----------------------------------------------------------------------------
# Crops
# -----------------------------------------------------------------------------

def gp_crop(ds=None) -> int:
    """Side of the centred crop the GP sees (its input is that crop, flattened)."""
    return min(cfg.GP_CROP, img_size(ds))


def gp_offset(ds=None) -> int:
    return (img_size(ds) - gp_crop(ds)) // 2


def center_crop(arr_nhwc, size):
    """Centre-crop an (N, H, W, C) stack to size x size."""
    off = (arr_nhwc.shape[1] - size) // 2
    return arr_nhwc[:, off:off + size, off:off + size, :]


def crop_origin(cx, cy, crop_size, H=None, W=None, ds=None):
    """Top-left corner of the crop around (cx, cy), clamped to the frame."""
    if H is None or W is None:
        H = W = img_size(ds)
    cs = int(crop_size)
    y0 = max(0, int(round(float(cy))) - cs // 2)
    if y0 + cs > H:
        y0 = H - cs
    x0 = max(0, int(round(float(cx))) - cs // 2)
    if x0 + cs > W:
        x0 = W - cs
    return max(0, y0), max(0, x0)


def crop_around_center(imgs, cx, cy, crop_size):
    """(crop_size x crop_size) window around (cx, cy), clamped to the frame.

    Shared by the LN model and by every LSTA routine, so the window is defined
    in exactly one place.
    """
    arr = np.asarray(imgs, dtype=np.float32)
    H, W = arr.shape[-2], arr.shape[-1]
    y0, x0 = crop_origin(cx, cy, crop_size, H, W)
    cs = int(crop_size)
    return arr[..., y0:y0 + cs, x0:x0 + cs]


# -----------------------------------------------------------------------------
# DataLoaders (transformer + CNN)
# -----------------------------------------------------------------------------

class RetinalDataset(TorchDataset):
    """Images and responses as tensors, one stimulus per item."""

    def __init__(self, imgs, resp):
        self.images = torch.tensor(imgs[:, np.newaxis], dtype=torch.float32)
        self.responses = torch.tensor(resp, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {'image': self.images[idx], 'response': self.responses[idx]}


def subset_indices(n, fraction, seed):
    """Reproducible random subset of ``n`` items, sorted."""
    rng = np.random.default_rng(seed)
    k = max(1, int(n * fraction))
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    return idx


def kfold_split(n, fold, n_folds=None, seed=None):
    """(train indices, held-out indices) of fold ``fold`` of a pool of ``n``.

    Classic K-fold: a seeded permutation of the pool is cut into ``n_folds``
    contiguous chunks of (nearly) equal size, chunk ``fold`` is held out and the
    rest is the training set. Every stimulus is held out in exactly one fold.
    The permutation is fixed (``cfg.KFOLD_SEED``), so the partition is identical
    for every model, every training seed and every run: two runs that differ
    only in their training seed are fitted on exactly the same stimuli, and the
    fold models of different architectures are compared on the same held-out
    stimuli. The dataset file holds one pool and never fixes this partition.
    """
    n_folds = cfg.N_FOLDS if n_folds is None else int(n_folds)
    seed = cfg.KFOLD_SEED if seed is None else seed
    fold = int(fold)
    if n_folds < 2:
        raise ValueError(f'K-fold cross-validation needs at least 2 folds, '
                         f'got n_folds={n_folds}')
    if not 0 <= fold < n_folds:
        raise ValueError(f'fold must be in [0, {n_folds}), got {fold}')
    if n < n_folds:
        raise ValueError(f'a pool of {n} stimuli cannot be cut into {n_folds} '
                         f'folds; the recording is too small')
    order = np.random.default_rng(seed).permutation(n)
    chunks = np.array_split(order, n_folds)
    held = chunks[fold]
    train = np.concatenate([c for k, c in enumerate(chunks) if k != fold])
    return np.sort(train), np.sort(held)


def fold_assignment(n, n_folds=None, seed=None):
    """(n,) the fold each pool stimulus is held out in."""
    n_folds = cfg.N_FOLDS if n_folds is None else int(n_folds)
    out = np.empty(n, dtype=np.int64)
    for k in range(n_folds):
        out[kfold_split(n, k, n_folds, seed)[1]] = k
    return out


def fold_indices(ds=None, fold=0, n_folds=None, data_fraction=1.0, seed=42):
    """(train indices, held-out indices) into the pool of a dataset.

    The fold is cut from the *whole* pool first, so the held-out stimuli of
    fold k are the same at every ``data_fraction``; the subsetting that the
    data-efficiency study sweeps is applied to the training folds only, drawn
    reproducibly from ``seed``. At ``data_fraction == 1`` no subsetting happens,
    so runs of the same fold that differ only in ``seed`` are fitted on an
    identical set of stimuli and differ only in initialisation, dropout and
    batch order.
    """
    i_tr, i_vl = kfold_split(meta(ds)['n_pool'], fold, n_folds)
    if data_fraction < 1.0:
        i_tr = i_tr[subset_indices(len(i_tr), data_fraction, seed)]
    return i_tr, i_vl


def get_dataloaders(ds=None, batch_size=32, data_fraction=1.0, random_seed=42,
                    fold=0, n_folds=None):
    """train / val / test loaders for one fold of the cross-validation.

    The pool is cut by `kfold_split`: fold ``fold`` of ``n_folds`` is the
    validation set (early stopping), the rest the training set. See
    `fold_indices` for what ``data_fraction`` and ``random_seed`` do.
    """
    d = load(ds)
    pool_imgs = d['images_pool'].squeeze(-1)
    pool_resp = d['responses_pool']

    i_tr, i_vl = fold_indices(ds, fold, n_folds, data_fraction, random_seed)
    images_test = d['images_test'].squeeze(-1)
    responses_test = d['responses_test'].mean(axis=0)

    return (
        DataLoader(RetinalDataset(pool_imgs[i_tr], pool_resp[i_tr]),
                   batch_size=batch_size, shuffle=True),
        DataLoader(RetinalDataset(pool_imgs[i_vl], pool_resp[i_vl]),
                   batch_size=batch_size, shuffle=False),
        DataLoader(RetinalDataset(images_test, responses_test),
                   batch_size=batch_size, shuffle=False),
    )


# -----------------------------------------------------------------------------
# GP inputs (flattened centre crops)
# -----------------------------------------------------------------------------

def gp_pool(ds=None, device='cpu', dtype=torch.float32):
    """(n_pool, crop^2) -- the pool the GP draws its training set from."""
    imgs = center_crop(load(ds)['images_pool'], gp_crop(ds))
    flat = imgs.reshape(imgs.shape[0], -1)
    return torch.tensor(flat, dtype=dtype).to(device)


def gp_pool_responses(ds=None, device='cpu', dtype=torch.float32):
    """(n_pool, N) responses aligned with ``gp_pool``."""
    return torch.tensor(load(ds)['responses_pool'], dtype=dtype).to(device)


def gp_test(ds=None, device='cpu', dtype=torch.float32):
    """(n_test, crop^2) test images and (n_rep, n_test, N) test responses."""
    d = load(ds)
    imgs = center_crop(d['images_test'], gp_crop(ds))
    X = torch.tensor(imgs.reshape(imgs.shape[0], -1), dtype=dtype).to(device)
    r = torch.tensor(d['responses_test'], dtype=dtype).to(device)
    return X, r


def gp_rf_centers(ds=None):
    """(N, 2) RF centres as normalised [-1, 1] coordinates of the GP crop."""
    cx, cy = centers(ds)
    off, side = gp_offset(ds), gp_crop(ds)
    eps_x = 2.0 * (np.asarray(cx) - off) / (side - 1) - 1.0
    eps_y = 2.0 * (np.asarray(cy) - off) / (side - 1) - 1.0
    return np.stack([eps_x, eps_y], axis=1)


def gp_pool_tolerance(X_pool, floor=1e-3):
    """Tolerance for a checkpoint's pool-sum integrity tag.

    float32 accumulation noise scales with the magnitude of the sum (a pool
    summing to ~9e5 is well past any absolute epsilon), while a real content
    mismatch shifts it by orders of magnitude more.
    """
    return max(floor, 1e-5 * abs(float(X_pool.sum().item())))
