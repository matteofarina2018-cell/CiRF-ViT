"""Band-limit the LSTA comparison to the frequencies the experiment resolves.

Why this exists
---------------
`lsta_corr` correlates a model's input gradient with the experimental LSTA over
the receptive-field window. Both maps carry energy at spatial frequencies where
the experiment measures only noise. That energy cannot contribute to the
covariance but does enter the model's variance in the denominator, so the raw
correlation penalises a model for having any content in a band where there is
nothing to agree with. The penalty is not neutral across models: the rougher a
model's gradients, the larger it is. Low-passing both sides removes the band
from the comparison instead of scoring models on it.

This is a change to the measurement, not to the models. It does not assert that
the suppressed structure is noise; on the model side it demonstrably is not
(see scripts/gradient_reproducibility.py). It only says the experiment cannot be used
to check it.

Choosing the cutoff
-------------------
``estimate_cutoff`` derives it from the experimental data alone, never from the
resulting correlations:

  signal window   the RF ellipse bounding box plus the same padding the
                  correlation uses, so the estimate covers the pixels that are
                  actually scored;
  noise window    a window of identical size placed far from the RF, where the
                  neuron does not respond, giving a per-frequency noise floor;
  cutoff          the highest frequency whose SNR still exceeds `snr_threshold`.

Spectra are estimated at the LSTA's native resolution. Estimating them after
resampling to the 108 px analysis frame would fold the interpolation's own
low-pass into the answer.

A Gaussian low-pass of standard deviation sigma has transfer exp(-2 pi^2
sigma^2 f^2), so its half-power point sits at f = 0.1874 / sigma, i.e. at a
spatial period of 5.34 * sigma. The cutoff period is converted to sigma through
that relation, after being expressed in 108 px units.

Result on the raw LSTAs, snr_threshold=2: sigma 2.08 (mouse) and 2.31
(salamander). DEFAULT_SIGMA is the common value used for both; re-derive it
with `python scripts/lsta_cutoff.py`.
"""

from __future__ import annotations

import numpy as np


# Half-power period of a Gaussian low-pass, in units of sigma:
# exp(-2 pi^2 sigma^2 f^2) = 1/2  ->  f = sqrt(ln2 / (2 pi^2)) / sigma
HALF_POWER_F = float(np.sqrt(np.log(2.0) / (2.0 * np.pi ** 2)))   # 0.18740
PERIOD_PER_SIGMA = 1.0 / HALF_POWER_F                             # 5.336

DEFAULT_SIGMA = 2.2       # px in the 108 frame; see module docstring
BBOX_PADDING = 8          # must match compute_lsta_corr_from_arrays
SNR_THRESHOLD = 2.0


def sigma_to_period(sigma):
    """Half-power spatial period (px) of a Gaussian low-pass."""
    return PERIOD_PER_SIGMA * float(sigma)


def period_to_sigma(period):
    """Gaussian sigma whose half-power point sits at `period` px."""
    return float(period) / PERIOD_PER_SIGMA


# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

def _kernel1d(sigma):
    r = max(1, int(round(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def lowpass(maps, sigma):
    """Gaussian low-pass over the last two axes of a (..., H, W) array.

    Separable, edge-replicating. `sigma <= 0` returns the input unchanged so
    callers can disable filtering without branching.
    """
    if sigma is None or sigma <= 0:
        return np.asarray(maps)
    a = np.asarray(maps, dtype=np.float64)
    k = _kernel1d(sigma)
    r = (len(k) - 1) // 2
    pad = [(0, 0)] * (a.ndim - 2) + [(r, r), (0, 0)]
    b = np.apply_along_axis(lambda m: np.convolve(m, k, 'valid'), -2,
                            np.pad(a, pad, mode='edge'))
    pad = [(0, 0)] * (a.ndim - 2) + [(0, 0), (r, r)]
    return np.apply_along_axis(lambda m: np.convolve(m, k, 'valid'), -1,
                               np.pad(b, pad, mode='edge'))


# ─────────────────────────────────────────────────────────────────────────────
# Cutoff estimation
# ─────────────────────────────────────────────────────────────────────────────

def _bboxes(ellipses, ellipse_scale, size, padding):
    out = []
    for n in range(len(ellipses)):
        ex = ellipses[n, 0, :] * ellipse_scale
        ey = ellipses[n, 1, :] * ellipse_scale
        x0 = int(max(0, np.floor(ex.min() - padding)))
        x1 = int(min(size, np.ceil(ex.max() + padding)))
        y0 = int(max(0, np.floor(ey.min() - padding)))
        y1 = int(min(size, np.ceil(ey.max() + padding)))
        out.append((y0, y1, x0, x1))
    return out


def _far_window(box, size):
    """Same-sized window pushed to the opposite side of the frame."""
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    ny0 = 0 if (y0 + y1) / 2 > size / 2 else size - h
    nx0 = 0 if (x0 + x1) / 2 > size / 2 else size - w
    ny0 = max(0, min(size - h, ny0))
    nx0 = max(0, min(size - w, nx0))
    return ny0, ny0 + h, nx0, nx0 + w


def _radial_power(patches, bins):
    """Mean normalised radial power spectrum over patches of differing sizes.

    The frequency axis is cycles per pixel, which is absolute, so patches of
    different sizes can be pooled.
    """
    tot = np.zeros(len(bins) - 1)
    cnt = np.zeros(len(bins) - 1)
    for a in patches:
        h, w = a.shape
        if h < 8 or w < 8:
            continue
        win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        p = np.abs(np.fft.fftshift(np.fft.fft2(a * win))) ** 2
        fy = np.fft.fftshift(np.fft.fftfreq(h))
        fx = np.fft.fftshift(np.fft.fftfreq(w))
        rad = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).ravel()
        idx = np.digitize(rad, bins) - 1
        pr = p.ravel() / (p.sum() + 1e-30)
        for i in range(len(bins) - 1):
            m = idx == i
            if m.any():
                tot[i] += pr[m].mean()
                cnt[i] += 1
    return tot / np.maximum(cnt, 1)


def estimate_cutoff(lsta, ellipses, ellipse_src, snr_threshold=SNR_THRESHOLD,
                    padding=BBOX_PADDING, n_bins=24, analysis_px=108,
                    return_profile=False):
    """Sigma (in `analysis_px` units) at which to low-pass the comparison.

    lsta        (N, N_img, S, S) experimental LSTA at its native resolution S
    ellipses    (N, 2, 360) RF outlines, in the `ellipse_src` pixel frame
    ellipse_src pixel frame the ellipses are stored in (72 mouse, 108 salamander)

    Returns the sigma, or (sigma, profile_dict) when `return_profile`.
    """
    lsta = np.asarray(lsta, dtype=np.float64)
    native = lsta.shape[-1]
    boxes = _bboxes(ellipses, native / float(ellipse_src), native, padding)

    sig, noi = [], []
    for n in range(lsta.shape[0]):
        y0, y1, x0, x1 = boxes[n]
        if y1 - y0 < 8 or x1 - x0 < 8:
            continue
        fy0, fy1, fx0, fx1 = _far_window(boxes[n], native)
        for i in range(lsta.shape[1]):
            sig.append(lsta[n, i, y0:y1, x0:x1])
            noi.append(lsta[n, i, fy0:fy1, fx0:fx1])
    if not sig:
        raise ValueError('no usable RF windows; check ellipses / ellipse_src')

    bins = np.linspace(0, 0.5, n_bins + 1)
    fc = (bins[:-1] + bins[1:]) / 2
    snr = _radial_power(sig, bins) / np.maximum(_radial_power(noi, bins), 1e-30)

    ok = np.where(snr > snr_threshold)[0]
    if len(ok) == 0:
        raise ValueError(f'SNR never exceeds {snr_threshold}')
    f_cut = fc[ok.max()]                       # cycles per native px
    period_native = 1.0 / f_cut
    period_analysis = period_native * analysis_px / native
    sigma = period_to_sigma(period_analysis)

    if not return_profile:
        return sigma
    return sigma, dict(freq_native=fc, snr=snr, native_px=native,
                       period_native=period_native,
                       period_analysis=period_analysis,
                       n_windows=len(sig),
                       median_box=float(np.median([b[1] - b[0] for b in boxes])))
