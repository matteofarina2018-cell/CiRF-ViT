#!/usr/bin/env python
"""How much of a model's gradient power the band limit removes.

    python scripts/band_power.py --dataset both
    python scripts/band_power.py --models transformer cnn GP

For each width sigma, reports the share of input-gradient power that sits above
the corresponding cutoff frequency f_c = 0.187 / sigma -- equivalently, on
spatial scales finer than 5.34 * sigma pixels. Measured inside the same
receptive-field window the LSTA comparison scores, averaged over every
independently trained run, neuron and stimulus.

This is what the LSTA benchmark cannot see. Two architectures can agree with the
experiment equally well inside the resolvable band and still differ sharply
above it; this table quantifies that difference instead of leaving it implicit.
The GP is included when asked for, but its kernel makes its gradients smooth by
construction, so the informative contrast is between the two gradient-trained
architectures.

Needs the fraction study at fraction 1.0 (scripts/fraction_study.py), whose runs
supply the independently trained models.
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import (add_dataset_arg, add_device_arg, banner, cfg,
                     datasets_from, device_from, require_lsta)
from cirf import data as dat
from cirf import lsta as L
from cirf.lsta_filter import HALF_POWER_F

#: same widths as the correlation table, so the two line up row by row
SIGMAS = (5.0, 3.0, 2.2, 2.0, 1.5, 1.0, 0.5)


def band_fraction(grads, cutoffs, boxes):
    """Mean over runs, neurons and images of P(f >= f_c) / P(total)."""
    N, I = grads.shape[1:3]
    acc = {fc: [] for fc in cutoffs}
    for n in range(N):
        y0, y1, x0, x1 = boxes[n]
        h, w = y1 - y0, x1 - x0
        if h < 8 or w < 8:
            continue
        window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        radius = np.sqrt(fy ** 2 + fx ** 2)
        patch = grads[:, n, :, y0:y1, x0:x1] * window
        power = np.abs(np.fft.fft2(patch, axes=(-2, -1))) ** 2
        total = power.sum(axis=(-2, -1))
        for fc in cutoffs:
            above = (power * (radius >= fc)).sum(axis=(-2, -1))
            acc[fc].append(above / np.maximum(total, 1e-30))
    return {fc: float(np.mean(acc[fc])) for fc in cutoffs}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(ap, default='both')
    add_device_arg(ap, default='cpu')
    ap.add_argument('--models', nargs='+', default=['transformer', 'cnn'],
                    choices=['transformer', 'cnn', 'GP'])
    ap.add_argument('--sigmas', type=float, nargs='+', default=list(SIGMAS))
    args = ap.parse_args()

    device = device_from(args, default='cpu')
    cutoffs = [0.0] + [HALF_POWER_F / s for s in args.sigmas]

    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        if not require_lsta(ds):
            continue
        banner(f'band power | {ds.label}')
        boxes = L.rf_bboxes(dat.lsta_reference(ds)['ellipses'], ds=ds)

        results = {}
        for m in args.models:
            grads, _ = L.run_gradients(m, ds, device)
            results[m] = band_fraction(grads.astype(np.float64), cutoffs, boxes)
            del grads

        print(f'\n{"sigma":>8}{"period (px)":>13}{"f_c (cyc/px)":>14}'
              + ''.join(f'{cfg.MODEL_ABBR[m]:>10}' for m in args.models))
        for sigma, fc in zip([None] + list(args.sigmas), cutoffs):
            label = 'none' if sigma is None else f'{sigma:g}'
            period = '-' if fc == 0 else f'{1 / fc:.1f}'
            print(f'{label:>8}{period:>13}{fc:>14.4f}'
                  + ''.join(f'{results[m][fc]:>10.3f}' for m in args.models))

        out = cfg.ensure(cfg.cache_dir(ds)) / 'band_power.npz'
        np.savez_compressed(out, cutoffs=np.array(cutoffs),
                            sigmas=np.array([np.nan] + list(args.sigmas)),
                            **{m: np.array([results[m][fc] for fc in cutoffs])
                               for m in args.models})
        print(f'  -> {out}')


if __name__ == '__main__':
    main()
