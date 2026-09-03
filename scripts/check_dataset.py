#!/usr/bin/env python
"""Validate a dataset file and report what the pipeline can do with it.

    python scripts/check_dataset.py --data my_recording.npz
    python scripts/check_dataset.py --dataset both

Checks the arrays against docs/DATA_FORMAT.md and prints the shapes, the
split-half reliability of every neuron (the ceiling any model can reach on it)
and which benchmarks the file supports. Run this first on a new recording: an
error here is a problem with the file, not with the models.
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import add_dataset_arg, banner, cfg, datasets_from
from cirf import data as dat
from cirf.metrics import c_eo_per_neuron


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(ap, default='mice')
    args = ap.parse_args()

    for ds in datasets_from(args):
        banner(f'checking {ds.npz}')
        try:
            meta = dat.validate(ds.npz)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(f'  INVALID: {exc}')
            continue

        cfg.set_dataset(ds)
        print(dat.describe(ds))

        rel = c_eo_per_neuron(dat.test_repeats(ds))
        weak = np.where(~(rel > 0))[0]
        print(f'\n  split-half reliability: median {np.median(rel):.3f}, '
              f'range [{np.nanmin(rel):.3f}, {np.nanmax(rel):.3f}]')
        if len(weak):
            print(f'  {len(weak)} neuron(s) with no reproducible signal '
                  f'(reliability <= 0): {list(map(int, weak))}')
            print('    every model scores 0 on those; consider dropping them')

        rates = dat.responses('pool', ds).mean(axis=0) / cfg.STIM_DURATION_S
        print(f'  mean firing rate: {rates.min():.2f} - {rates.max():.2f} Hz '
              f'(median {np.median(rates):.2f})')

        print('\n  benchmarks available:')
        print('    prediction (EV, adjusted R^2)            yes')
        print(f'    LSTA comparison                          '
              f'{"yes" if meta["has_lsta"] and meta["has_ellipses"] else "no"}')
        print('    data-efficiency sweep                    yes')
        print('\n  next: python scripts/train.py --model transformer '
              f'--data {ds.npz.name}')


if __name__ == '__main__':
    main()
