#!/usr/bin/env python
"""Derive the band limit of the LSTA comparison from the experimental data.

    python scripts/lsta_cutoff.py --dataset both
    python scripts/lsta_cutoff.py --data my.npz --snr 3

The LSTA is estimated from a finite number of spikes, so above some spatial
frequency it measures only noise. Correlating a model against it there cannot
raise the covariance but does inflate the model's variance in the denominator,
which penalises a model for having content in a band where there is nothing to
agree with -- and penalises the roughest model most. The comparison is therefore
low-passed on both sides first.

The cutoff comes from the experiment alone, never from the resulting
correlations: for every neuron, the radial power spectrum inside its
receptive-field window is compared with the spectrum of an identical window
placed far from the receptive field, where the neuron does not respond. The
usable band is where that ratio exceeds `--snr`; the Gaussian whose half-power
point sits at the cutoff gives the sigma.

Spectra are estimated at the LSTA's native resolution: doing it after resampling
would fold the interpolation's own low-pass into the answer.

This is a change to the measurement, not to the models. It does not claim the
suppressed structure is noise on the model side -- see
scripts/gradient_reproducibility.py, which shows it is not.
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import add_dataset_arg, banner, cfg, datasets_from, require_lsta
from cirf import data as dat
from cirf import lsta_filter


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(ap, default='both')
    ap.add_argument('--snr', type=float, default=lsta_filter.SNR_THRESHOLD,
                    help=f'signal-to-noise threshold defining the usable band '
                         f'(default {lsta_filter.SNR_THRESHOLD})')
    args = ap.parse_args()

    sigmas = []
    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        if not require_lsta(ds):
            continue
        banner(f'LSTA band limit | {ds.label} | SNR > {args.snr}')
        ref = dat.lsta_reference(ds)
        sigma, prof = lsta_filter.estimate_cutoff(
            ref['lsta'], ref['ellipses'], dat.ellipse_frame(ds),
            snr_threshold=args.snr, analysis_px=dat.img_size(ds),
            return_profile=True)
        sigmas.append(sigma)

        print(f'  native {prof["native_px"]} px, {prof["n_windows"]} windows, '
              f'median RF box {prof["median_box"]:.0f} px')
        print(f'  {"period (native px)":>20}{"period (analysis px)":>22}{"SNR":>8}')
        for f, snr in zip(prof['freq_native'], prof['snr']):
            if 0.06 <= f <= 0.42:
                print(f'  {1 / f:>20.1f}'
                      f'{dat.img_size(ds) / prof["native_px"] / f:>22.1f}{snr:>8.2f}')
        print(f'  cutoff: {prof["period_native"]:.1f} px native = '
              f'{prof["period_analysis"]:.1f} px in the analysis frame'
              f'  ->  sigma = {sigma:.2f}')

    if len(sigmas) > 1:
        print(f'\nmean over datasets: sigma = {np.mean(sigmas):.2f}')
    if sigmas:
        print(f'value in use (cirf.lsta_filter.DEFAULT_SIGMA): '
              f'{lsta_filter.DEFAULT_SIGMA}')


if __name__ == '__main__':
    main()
