"""Local spike-triggered average: model gradients, and their comparison.

The LSTA is measured by superimposing a low-contrast binary checkerboard on a
fixed natural image and spike-triggered-averaging the noise component. Because
the noise is a small perturbation around that image, the LSTA estimates the
neuron's first-order sensitivity to the stimulus *at* that image -- the
experimental counterpart of the gradient of a model's predicted rate with
respect to its input. No model is ever trained on it, so comparing the two is a
zero-shot functional test, and any model with a differentiable input can take
it.

This module owns three things:

  * the input gradients of each architecture (`gradients`),
  * the post-processing applied identically to model and experiment,
  * the per-neuron correlation itself (`corr_from_arrays`), band-limited to the
    frequencies the experiment resolves -- see `cirf.lsta_filter`.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage

from . import config as cfg
from . import data as dat
from . import lsta_filter

PADDING = 8            # px added around the RF ellipse before correlating
EXPON = 2              # signed power applied to both maps
VMAX_THRESH = 1.0
PUT_TO0_FRAC = 0.2     # zero everything below this share of the map maximum


# -----------------------------------------------------------------------------
# Map handling
# -----------------------------------------------------------------------------

def resize_2d(arr, size):
    """Bilinear resampling of a single 2D map to size x size."""
    pil = PILImage.fromarray(np.asarray(arr, dtype=np.float32), mode='F')
    return np.array(pil.resize((size, size), PILImage.BILINEAR))


def postprocess(arr, expon=EXPON, vmax_thresh=VMAX_THRESH,
                put_to0_frac=PUT_TO0_FRAC):
    """Signed power + threshold, applied identically to model and experiment.

    Returns (processed map, vmax) where vmax is the symmetric colour limit.
    """
    arr = np.asarray(arr, dtype=float).copy()
    arr = (np.sign(arr) * np.exp(np.log(np.abs(arr) + 1e-12) * expon)
           if expon % 2 == 0 else arr ** expon)
    vmax = max(arr.max(), -arr.min()) * vmax_thresh
    if vmax > 0:
        arr[np.abs(arr) < vmax * put_to0_frac] = 0.0
    return arr, vmax


def rf_bboxes(ellipses, padding=PADDING, size=None, offset=0, ds=None):
    """Per-neuron (y0, y1, x0, x1) RF boxes in the analysis frame.

    The outlines are stored in their own pixel frame (`data.ellipse_frame`) and
    are scaled here. `offset` shifts them into the GP's smaller crop.
    """
    if size is None:
        size = dat.img_size(ds)
    scale = dat.img_size(ds) / dat.ellipse_frame(ds)
    boxes = []
    for n in range(len(ellipses)):
        ex = ellipses[n, 0, :] * scale - offset
        ey = ellipses[n, 1, :] * scale - offset
        boxes.append((int(max(0, ey.min() - padding)), int(min(size, ey.max() + padding)),
                      int(max(0, ex.min() - padding)), int(min(size, ex.max() + padding))))
    return boxes


def to_analysis_frame(arr, ds=None):
    """Bring one map into the image frame (pad the GP crop, resample the rest)."""
    arr = np.asarray(arr)
    P, crop, off = dat.img_size(ds), dat.gp_crop(ds), dat.gp_offset(ds)
    if arr.shape[-1] == crop and crop != P:
        pad = np.zeros((P, P), dtype=np.float32)
        pad[off:off + crop, off:off + crop] = arr
        return pad
    if arr.shape[-1] != P:
        return resize_2d(arr, P)
    return arr


def prepare_panel(arr, lowpass_sigma=None, ds=None):
    """One map, ready to display exactly as the metric consumes it.

    Band-limiting matters here too: without it a panel would show detail at
    frequencies the correlation never scores. Pass 0 to see the raw map.
    """
    if lowpass_sigma is None:
        lowpass_sigma = lsta_filter.DEFAULT_SIGMA
    arr = to_analysis_frame(arr, ds)
    if lowpass_sigma > 0:
        arr = lsta_filter.lowpass(arr, lowpass_sigma)
    p, v = postprocess(arr)
    return p, (v if v > 0 else 1.0)


# -----------------------------------------------------------------------------
# The metric
# -----------------------------------------------------------------------------

def corr_from_arrays(lsta_model, lsta_exp, ellipses, model_size=None,
                     padding=PADDING, expon=EXPON, vmax_thresh=VMAX_THRESH,
                     put_to0_frac=PUT_TO0_FRAC, lowpass_sigma=None, ds=None):
    """(N,) correlation between model gradients and the experimental LSTA.

    lsta_model : (N, n_img, model_size, model_size)
    lsta_exp   : (N, n_img, S, S) at the LSTA's native resolution, resampled here
    ellipses   : (N, 2, n_points) RF outlines, in the dataset's ellipse frame
    model_size : side of the model maps (the image size, or the GP crop)

    Both sides are low-passed with the same Gaussian before the comparison
    (`lowpass_sigma=None` uses `lsta_filter.DEFAULT_SIGMA`; pass 0 for the
    unfiltered metric), then post-processed, then correlated inside the RF box.
    """
    if lowpass_sigma is None:
        lowpass_sigma = lsta_filter.DEFAULT_SIGMA
    if model_size is None:
        model_size = int(lsta_model.shape[-1])
    P, crop, off = dat.img_size(ds), dat.gp_crop(ds), dat.gp_offset(ds)
    on_gp_crop = (model_size == crop and crop != P)
    boxes = rf_bboxes(ellipses, padding=padding, size=model_size,
                      offset=off if on_gp_crop else 0, ds=ds)

    N, n_img = lsta_model.shape[:2]
    out = np.zeros(N, dtype=np.float32)
    for n in range(N):
        y0, y1, x0, x1 = boxes[n]
        scores = []
        for i in range(n_img):
            exp_map = resize_2d(lsta_exp[n, i], P)
            if on_gp_crop:
                exp_map = exp_map[off:off + crop, off:off + crop]
            mod_map = lsta_model[n, i]
            if lowpass_sigma > 0:
                exp_map = lsta_filter.lowpass(exp_map, lowpass_sigma)
                mod_map = lsta_filter.lowpass(mod_map, lowpass_sigma)
            pe, _ = postprocess(exp_map, expon, vmax_thresh, put_to0_frac)
            pm, _ = postprocess(mod_map, expon, vmax_thresh, put_to0_frac)
            scores.append(correlation(pe[y0:y1, x0:x1], pm[y0:y1, x0:x1]))
        out[n] = float(np.mean(scores)) if scores else 0.0
    return out


def correlation(a, b):
    """Pearson correlation of two flattened maps; 0.0 if either is constant."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or a.std() <= 0 or b.std() <= 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# -----------------------------------------------------------------------------
# Model gradients
#
# One entry point, `gradients()`, with an implementation per architecture.
# Every implementation rebuilds the autograd graph for each (image, neuron)
# pair. That is deliberate: back-propagating N times through a single retained
# graph silently corrupts the CNN's maps (its BatchNorm layers do not survive
# repeated backward passes), and the cost is only a few extra forwards.
# -----------------------------------------------------------------------------

def gradients(model_type, model, base_imgs, device='cpu', n_cells=None, ds=None):
    """(N, n_img, P, P) input gradients of the predicted rate."""
    if model_type == 'cnn':
        return _cnn_gradients(model, base_imgs, device, n_cells, ds)
    return _torch_gradients(model, base_imgs, device, n_cells, ds)


def _n_cells_of(model, n_cells):
    if n_cells is not None:
        return int(n_cells)
    if hasattr(model, 'n_cells'):
        return int(model.n_cells)
    return int(model.readout.bias.shape[0])


def _torch_gradients(model, base_imgs, device, n_cells=None, ds=None):
    """Generic autograd path (transformer and any other pixel-space model)."""
    model.eval()
    N = _n_cells_of(model, n_cells)
    P = int(np.asarray(base_imgs).shape[-1])
    out = np.zeros((N, len(base_imgs), P, P), dtype=np.float32)
    for j in range(len(base_imgs)):
        for n in range(N):
            x = torch.tensor(base_imgs[j][None, None], dtype=torch.float32,
                             device=device, requires_grad=True)
            model(x)[0, n].backward()
            out[n, j] = x.grad[0, 0].detach().cpu().numpy()
    return out


def _cnn_gradients(model, base_imgs, device, n_cells=None, ds=None):
    """Explicit chain rule through the CNN readout.

    With y_n = softplus(sum_c v[n,c] sum_hw u[n,h,w] f[c,h,w] + bias[n]),

        dy_n / df[c,h,w] = sigmoid(h_n) * v[n,c] * u[n,h,w]

    which is pushed through the convolutional core by a weighted backward.
    Doing it this way instead of calling backward on the assembled model was
    validated against the published maps at r = 1.00000, scale 1.000, for every
    neuron and image of both datasets.
    """
    model.eval()
    N = _n_cells_of(model, n_cells)
    P = int(np.asarray(base_imgs).shape[-1])
    u_all, v_all = model.readout.u.detach(), model.readout.v.detach()
    out = np.zeros((N, len(base_imgs), P, P), dtype=np.float32)
    for j in range(len(base_imgs)):
        for n in range(N):
            x = torch.tensor(base_imgs[j][None, None], dtype=torch.float32,
                             device=device, requires_grad=True)
            f = model.core(x)
            with torch.no_grad():
                h = (torch.einsum('bchw,nhw->bnc', f, u_all)
                     * v_all).sum(-1) + model.readout.bias
                sig = torch.sigmoid(h[0, n])
            weight = (u_all[n][None, :, :] * v_all[n][:, None, None])[None] * sig
            f.backward(gradient=weight)
            out[n, j] = x.grad[0, 0].detach().cpu().numpy()
    return out


def gp_gradients(models, base_imgs, device='cpu', ds=None):
    """(N, n_img, crop, crop) GP gradients from reconstructed model objects.

    The latent function is lambda(x) = ((K(x, X_tilde) B) / eigvals_b) m_b; its
    gradient with respect to the flattened crop is the GP's LSTA.
    """
    crop, off = dat.gp_crop(ds), dat.gp_offset(ds)
    patch = np.asarray(base_imgs)[:, off:off + crop, off:off + crop]
    flat = torch.tensor(patch.reshape(len(patch), -1), dtype=torch.float32)
    out = np.zeros((len(models), len(base_imgs), crop, crop), dtype=np.float32)
    for n, m in enumerate(models):
        m.eval()
        for j in range(len(base_imgs)):
            x = flat[j:j + 1].clone().to(device).requires_grad_(True)
            _gp_latent(m, x).backward()
            out[n, j] = x.grad[0].detach().cpu().numpy().reshape(crop, crop)
    return out


def _gp_latent(model, x_flat):
    K = model.kernel(x_flat, model.X_tilde).to_dense()
    a = (K @ model.state.B) / model.state.eigvals_b.unsqueeze(0)
    return (a @ model.state.m_b).squeeze()


def gp_gradients_from_dir(ckpt_dir, base_imgs, device='cpu', ds=None):
    """(N, n_img, crop, crop) GP gradients straight from the saved tensors.

    Bypasses the checkpoint loader, which rebuilds the eigenbasis at load time
    and pads m_b when the recomputed rank differs from the saved one -- fine for
    the canonical checkpoints, a size mismatch on some fraction-study ones.
    Everything the latent function needs is stored, so it is evaluated directly.
    Verified against the canonical maps at a median correlation of 1.0000.
    """
    from models.gp.kernels import create_kernel

    from . import checkpoints as _ckpt

    X_pool = dat.gp_pool(ds, device)
    crop, off = dat.gp_crop(ds), dat.gp_offset(ds)
    patch = np.asarray(base_imgs)[:, off:off + crop, off:off + crop]
    out = []
    for f in _ckpt.cell_paths(ckpt_dir):
        ck = torch.load(f, map_location=device, weights_only=False)
        ks, md, conf = ck['kernel_state_dict'], ck['metadata'], ck['config']
        dtype = ck['m_b'].dtype
        kernel = create_kernel(conf, md['n_px_side'],
                               ks['eps_0x'].item(), ks['eps_0y'].item())
        kernel.load_state_dict(ks)
        kernel = kernel.to(device=device, dtype=dtype)
        X_train = X_pool[ck['pool_indices']].to(dtype)
        X_tilde = X_train[:md.get('M', X_train.shape[0])]
        B, eig, m_b = ck['B'].to(dtype), ck['eigvals_b'].to(dtype), ck['m_b'].to(dtype)
        flat = torch.tensor(patch.reshape(len(patch), -1)).to(dtype)
        maps = []
        for j in range(len(flat)):
            x = flat[j:j + 1].clone().requires_grad_(True)
            (((kernel(x, X_tilde).to_dense() @ B) / eig.unsqueeze(0)) @ m_b).squeeze().backward()
            maps.append(x.grad[0].detach().cpu().numpy().reshape(crop, crop))
        out.append(np.stack(maps))
    return np.stack(out).astype(np.float32)


def run_gradients(model_type, ds=None, device='cpu', frac=1.0, folds=None,
                  seeds=None, verbose=True):
    """(n_runs, N, n_img, P, P) gradients of several independently trained runs.

    Reads the fraction-study checkpoints at `frac` (default: the runs trained on
    all the training folds), one run per (fold, seed): runs of the same fold
    differ only in initialisation, dropout and batch order, runs of different
    folds also in which stimuli were held out. Used by the analyses that ask
    whether the structure in a model's gradients is a property of the
    architecture or of the particular training run.
    """
    from . import checkpoints as ckpt
    from . import config as _cfg

    dataset = _cfg.resolve(ds or _cfg.active())
    base_imgs = dat.lsta_base_images(dataset)
    folds = list(folds if folds is not None else _cfg.FOLDS)
    seeds = list(seeds if seeds is not None else _cfg.FRAC_SEEDS)
    runs = [(k, s) for k in folds for s in seeds]

    out, names = [], []
    for k, s in runs:
        d = _cfg.fraction_dir(model_type, frac, k, s, dataset)
        if not ckpt.exists(model_type, d):
            continue
        if model_type == 'GP':
            maps = gp_to_analysis_frame(
                gp_gradients_from_dir(d, base_imgs, device, dataset), dataset)
        else:
            model, _ = ckpt.load_torch_model(model_type, d, dataset, device)
            maps = gradients(model_type, model, base_imgs, device,
                             n_cells=dat.n_cells(dataset), ds=dataset)
        out.append(maps)
        names.append(f'fold_{k}_seed_{s}')
        if verbose:
            print(f'  {model_type:>11} fold {k} seed {s} ({len(out)}/{len(runs)})  '
                  f'|grad|max = {np.abs(maps).max():.3e}', flush=True)
    if len(out) < 2:
        raise SystemExit(
            f'need at least 2 runs of {model_type} at fraction {frac:.0%}; '
            f'found {len(out)} under {_cfg.fraction_root(model_type, dataset)}')
    return np.stack(out), names


def gaussian_blur(maps, sigma):
    """Gaussian blur over the last two axes (the smooth part of a map)."""
    return lsta_filter.lowpass(maps, sigma)


def highpass(maps, sigma):
    """The fine structure: a map minus its Gaussian-blurred version."""
    if sigma is None or sigma <= 0:
        return np.asarray(maps)
    return np.asarray(maps) - lsta_filter.lowpass(maps, sigma)


def gp_to_analysis_frame(maps_crop, ds=None):
    """(N, n_img, crop, crop) -> (N, n_img, P, P), zero outside the crop."""
    P, crop, off = dat.img_size(ds), dat.gp_crop(ds), dat.gp_offset(ds)
    out = np.zeros((*maps_crop.shape[:2], P, P), dtype=np.float32)
    out[:, :, off:off + crop, off:off + crop] = maps_crop
    return out


def ln_gradients(cells, base_imgs, crop_size, ds=None, device='cpu',
                 per_seed=False):
    """Gradients of the per-cell LN models.

    `cells` is the per-cell list of models returned by
    `cirf.checkpoints.load_ln_models`: cells[c] holds that cell's repeats. Each
    LN only sees its own crop, so its gradient is zero everywhere else and is
    written back into the full frame.

    Returns (N, n_img, P, P) averaged over repeats, or
    (n_seeds, N, n_img, P, P) with `per_seed=True` (the figure that scores the
    LN correlates each repeat separately, then averages the scores).
    """
    cx_all, cy_all = dat.centers(ds)
    imgs = np.asarray(base_imgs, dtype=np.float32)
    P = imgs.shape[-1]
    n_seeds = max(len(s) for s in cells)
    out = np.zeros((n_seeds, len(cells), len(imgs), P, P), dtype=np.float32)
    cs = int(crop_size)
    for c, seeds in enumerate(cells):
        y0, x0 = dat.crop_origin(cx_all[c], cy_all[c], cs, P, P)
        for k, m in enumerate(seeds):
            x = torch.tensor(imgs, dtype=torch.float32, device=device,
                             requires_grad=True)
            m(x[:, y0:y0 + cs, x0:x0 + cs]).sum().backward()
            out[k, c] = x.grad.detach().cpu().numpy()
    return out if per_seed else out.mean(axis=0)
