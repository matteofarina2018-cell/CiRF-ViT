"""CiRF-ViT: a vision transformer with a per-neuron receptive-field crop, and
the benchmark suite it is evaluated with.

Typical use on your own recording (see docs/DATA_FORMAT.md for the file layout):

    from cirf import config, data, evaluate
    from models.transformer import train

    ds = config.register_path('my_recording.npz', label='my retina')
    print(data.describe(ds))
    train(ds)                                   # fit CiRF-ViT
    evaluate.evaluate_model('transformer', ds=ds)   # score it

The command-line equivalents live in scripts/ (`train.py`, `evaluate.py`,
`fraction_study.py`, ...), all of which accept `--data my_recording.npz`.

Modules
-------
config       paths, dataset registry, palette
data         dataset loading, crops, dataloaders
metrics      explained variance and adjusted R^2, per neuron
lsta         model gradients and the LSTA comparison
lsta_filter  the band limit that comparison is restricted to
checkpoints  rebuilding trained models from disk
evaluate     benchmarks: eval_data.npz, data-efficiency sweep
plots        figure styles and recurring panels
"""

__version__ = '1.0.0'

from . import config, data, metrics, lsta, lsta_filter, checkpoints, evaluate, plots

__all__ = ['config', 'data', 'metrics', 'lsta', 'lsta_filter', 'checkpoints',
           'evaluate', 'plots', '__version__']
