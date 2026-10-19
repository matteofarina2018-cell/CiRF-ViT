#!/usr/bin/env python
"""Test metrics of every data-efficiency run that has none yet.

    python scripts/score_fractions.py --dataset mice

Walks results/<dataset>/fraction_study/{transformer,cnn}/frac_XXX/fold_K/seed_S/
and writes metrics.json (EV, adjusted R^2, noise-corrected correlation per
neuron) next to every checkpoint that lacks one. The plain-ViT runs write their
own (scripts/fraction_study_plain_vit.py).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from _common import add_dataset_arg, add_device_arg, cfg, datasets_from, device_from
from cirf import data as dat
from cirf.evaluate import predict
from cirf.metrics import (adjusted_r2_per_neuron, cc_norm_per_neuron,
                          explained_variance_per_neuron)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_dataset_arg(ap)
    add_device_arg(ap)
    args = ap.parse_args()
    device = device_from(args)
    for ds in datasets_from(args):
        cfg.set_dataset(ds)
        rep = dat.test_repeats(ds)
        for model_type in ('transformer', 'cnn'):
            root = cfg.fraction_root(model_type, ds)
            for d in sorted(root.glob('frac_*/fold_*/seed_*')):
                if (d / 'metrics.json').exists() or not (d / 'model.pt').exists():
                    continue
                y = predict(model_type, d, ds, device)
                m = {'dataset': ds.key, 'model': model_type,
                     'fraction': int(d.parent.parent.name[5:]) / 100,
                     'fold': int(d.parent.name[5:]), 'seed': int(d.name[5:]),
                     'ev': explained_variance_per_neuron(y, rep).tolist(),
                     'adj_r2': adjusted_r2_per_neuron(y, rep).tolist(),
                     'cc_norm': cc_norm_per_neuron(y, rep).tolist()}
                (d / 'metrics.json').write_text(json.dumps(m), encoding='utf-8')
                print(f'{ds.key} {model_type} {d.parent.parent.name} '
                      f'{d.parent.name}: EV {np.mean(m["ev"]):.4f}', flush=True)


if __name__ == '__main__':
    main()
