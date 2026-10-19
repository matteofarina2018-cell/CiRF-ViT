"""Response-prediction metrics (Goldin et al., PNAS 2023, Eq. 5).

Both metrics are computed per neuron against the *repeated* test responses, so
they are normalised by how reproducible the neuron itself is. The repeats are
split into even and odd halves; ``c_eo`` is the correlation between the two
half-averages, i.e. an estimate of the ceiling any model could reach.

    explained variance  EV     = mean(corr(pred, even), corr(pred, odd)) / c_eo
    adjusted R^2        adjR2  = mean(corr(pred, even), corr(pred, odd))^2 / c_eo

A neuron whose halves anti-correlate (c_eo <= 0) carries no reproducible signal
and scores 0, as does a model whose prediction is constant.

Note that the normalisation is not calibrated to exactly 1: an oracle that
predicts the mean of the repeats scores about 1.05-1.08 on the paper's two
recordings, and more on individual neurons. Values slightly above 1 are
therefore expected and are not evidence of a leak.
"""

from __future__ import annotations

import numpy as np


def _half_averages(y_true_repeated):
    even = list(range(0, y_true_repeated.shape[1], 2))
    odd = list(range(1, y_true_repeated.shape[1], 2))
    return (y_true_repeated[:, even, :].mean(axis=1),
            y_true_repeated[:, odd, :].mean(axis=1))


def _paired_correlations(y_pred, y_true_repeated):
    """Yields (i, c_even_odd, c_pred_even, c_pred_odd) for scorable neurons."""
    r_even, r_odd = _half_averages(y_true_repeated)
    for i in range(y_pred.shape[1]):
        c_eo = np.corrcoef(r_even[:, i], r_odd[:, i])[0, 1]
        if not (c_eo > 0):
            continue
        c_pe = np.corrcoef(y_pred[:, i], r_even[:, i])[0, 1]
        c_po = np.corrcoef(y_pred[:, i], r_odd[:, i])[0, 1]
        if np.isnan(c_pe) or np.isnan(c_po):   # constant prediction
            continue
        yield i, c_eo, c_pe, c_po


def explained_variance_per_neuron(y_pred, y_true_repeated):
    """(N,) explained variance of the predicted rate, per neuron."""
    out = np.zeros(y_pred.shape[1])
    for i, c_eo, c_pe, c_po in _paired_correlations(y_pred, y_true_repeated):
        out[i] = max((c_pe + c_po) / 2.0 / c_eo, 0.0)
    return out


def adjusted_r2_per_neuron(y_pred, y_true_repeated):
    """(N,) adjusted R^2 of the predicted rate, per neuron."""
    out = np.zeros(y_pred.shape[1])
    for i, c_eo, c_pe, c_po in _paired_correlations(y_pred, y_true_repeated):
        out[i] = max(((c_pe + c_po) / 2.0) ** 2 / c_eo, 0.0)
    return out


def cc_norm_per_neuron(y_pred, y_true_repeated):
    """(N,) noise-corrected correlation: the EV with sqrt(c_eo).

    mean(corr(pred, even), corr(pred, odd)) / sqrt(c_eo), which an oracle
    scores 1 on (Schoppe et al., 2016). Equals sqrt(adjR2) where the mean
    correlation is positive.
    """
    out = np.zeros(y_pred.shape[1])
    for i, c_eo, c_pe, c_po in _paired_correlations(y_pred, y_true_repeated):
        out[i] = max((c_pe + c_po) / 2.0 / np.sqrt(c_eo), 0.0)
    return out


METRIC_FNS = {'ev': explained_variance_per_neuron,
              'adj_r2': adjusted_r2_per_neuron}
METRIC_TITLES = {'ev': 'Explained Variance', 'adj_r2': 'Adjusted R²',
                 'lsta_corr': 'LSTA Correlation'}
#: key each metric is stored under inside eval_data.npz
NPZ_KEY = {'ev': 'explained_variance', 'adj_r2': 'adjusted_r2',
           'lsta_corr': 'lsta_corr'}


def c_eo_per_neuron(y_true_repeated):
    """(N,) split-half reliability -- a property of the data, not of a model."""
    r_even, r_odd = _half_averages(y_true_repeated)
    return np.array([np.corrcoef(r_even[:, i], r_odd[:, i])[0, 1]
                     for i in range(y_true_repeated.shape[2])])


def adj_r2_to_ev(adj, c_eo):
    """EV from adjusted R^2, using the same c_eo both were normalised by.

    adjR2 = m^2 / c_eo and EV = m / c_eo with m the mean half-correlation, so
    EV = sqrt(adjR2 / c_eo). Used where an experiment stored adjusted R^2 but
    the figure reports EV.
    """
    adj = np.asarray(adj, dtype=float)
    c_eo = np.asarray(c_eo, dtype=float)
    return np.where((adj > 0) & (c_eo > 0),
                    np.sqrt(adj / np.maximum(c_eo, 1e-12)), 0.0)
