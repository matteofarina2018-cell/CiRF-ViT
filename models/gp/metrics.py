"""
Shared Evaluation Metrics for GP Models

This module contains evaluation metrics used by both eigenspace and default_gpy
implementations. These are pure functions with no mode-specific dependencies.

Functions:
- compute_r_squared: R² coefficient of determination
- compute_pearson_correlation: Pearson correlation coefficient
- compute_explained_variance: Explained variance (correlation ratio, NOT the paper's adjusted R²)
- compute_adjusted_r_squared: Adjusted R² from Goldin et al. 2023 PNAS, Eq. 5

Extracted from train.py during codebase reorganization (2025-02).
"""

import torch


def compute_r_squared(y_true, y_pred):
    """Compute R² (coefficient of determination).

    Args:
        y_true: True values, shape (n,) or (n_repeats, n)
        y_pred: Predicted values, shape (n,)

    Returns:
        R² value (scalar)
    """
    if y_true.ndim == 2:
        # Average over repeats
        y_true = y_true.mean(dim=0)

    y_true = y_true.float()
    y_pred = y_pred.float()

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()

    r2 = 1 - ss_res / ss_tot

    return r2.item()


def compute_pearson_correlation(y_true, y_pred):
    """Compute Pearson correlation coefficient.

    Args:
        y_true: True values, shape (n,) or (n_repeats, n)
        y_pred: Predicted values, shape (n,)

    Returns:
        Pearson correlation (scalar)
    """
    if y_true.ndim == 2:
        # Average over repeats
        y_true = y_true.mean(dim=0)

    y_true = y_true.float()
    y_pred = y_pred.float()

    # Center the values
    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()

    # Compute correlation
    numerator = (y_true_centered * y_pred_centered).sum()
    denominator = torch.sqrt((y_true_centered ** 2).sum() * (y_pred_centered ** 2).sum())

    corr = numerator / denominator

    return corr.item()


def compute_spearman_correlation(y_true, y_pred):
    """Compute Spearman rank correlation coefficient.

    Pearson correlation applied to the ranks of the values.
    Invariant to monotonic transformations of predictions.

    Args:
        y_true: True values, shape (n,) or (n_repeats, n)
        y_pred: Predicted values, shape (n,)

    Returns:
        Spearman correlation (scalar)
    """
    from scipy.stats import spearmanr

    if y_true.ndim == 2:
        y_true = y_true.mean(dim=0)

    result = spearmanr(
        y_true.detach().cpu().float().numpy(),
        y_pred.detach().cpu().float().numpy()
    )
    return float(result.correlation)


def compute_explained_variance(r_test, f_pred):
    """Compute explained variance normalized by cell reliability.

    NOTE: This is NOT the adjusted R² from Goldin et al. 2023 (Eq. 5).
    Use compute_adjusted_r_squared() for that metric.

    explained_var = mean_accuracy / reliability

    This is a correlation ratio: it divides by reliability directly (not
    sqrt(reliability)) and is not squared. Relationship to adjusted R²:
        adjusted_r2 = explained_var² * reliability

    Matches the reference implementation in varGP utils.py:explained_variance().

    Args:
        r_test: Test responses with repetitions, shape (n_repeats, n_images)
        f_pred: Predicted firing rates, shape (n_images,)

    Returns:
        explained_var: Fraction of explainable correlation captured (scalar)
        reliability: Cell reliability = |corr(r_even, r_odd)| (scalar)
    """
    r_test = r_test.float()
    f_pred = f_pred.float()

    # Split into even and odd repetitions
    r_even = r_test[0::2, :].mean(dim=0)  # (n_images,)
    r_odd = r_test[1::2, :].mean(dim=0)   # (n_images,)

    # Reliability = correlation between even and odd halves
    reliability = torch.corrcoef(torch.stack([r_even, r_odd]))[0, 1].abs()

    # Accuracy = average correlation with each half
    accuracy_even = torch.corrcoef(torch.stack([f_pred, r_even]))[0, 1]
    accuracy_odd = torch.corrcoef(torch.stack([f_pred, r_odd]))[0, 1]
    accuracy = 0.5 * (accuracy_even + accuracy_odd)

    # Explained variance = accuracy / reliability
    explained_var = accuracy / reliability

    return explained_var.item(), reliability.item()


def compute_adjusted_r_squared(r_test, f_pred):
    """Adjusted R² from Goldin et al. 2023 PNAS, Eq. 5.

    Reference: "Scalable Gaussian process inference of neural responses to
    natural images", Goldin et al., PNAS 2023 (doi:10.1073/pnas.2301150120).

    Adjusted R² = (mean_accuracy / sqrt(reliability))²
                = mean_accuracy² / reliability

    where:
        mean_accuracy = 0.5 * (corr(f_pred, r_even) + corr(f_pred, r_odd))
        reliability   = corr(r_even, r_odd)

    Based on the Spearman-Brown correction: the observed correlation between
    prediction and noisy response is attenuated by sqrt(reliability). Dividing
    by sqrt(reliability) recovers the correlation with the true (noise-free)
    signal. Squaring gives the fraction of true signal variance explained.

    A value of 1.0 means the model explains all signal variance given the
    noise ceiling. Always >= 0 by construction (squared ratio).

    This differs from compute_explained_variance(), which returns a correlation
    ratio (not squared) and divides by reliability (not sqrt):
        adjusted_r2 = explained_var² * reliability

    Args:
        r_test: Test responses with repetitions, shape (n_repeats, n_images)
        f_pred: Predicted firing rates, shape (n_images,)

    Returns:
        adjusted_r2: Adjusted R² value (scalar)
    """
    r_test = r_test.float()
    f_pred = f_pred.float()

    # Split into even and odd repetitions
    r_even = r_test[0::2, :].mean(dim=0)
    r_odd = r_test[1::2, :].mean(dim=0)

    # Reliability = correlation between even and odd halves
    reliability = torch.corrcoef(torch.stack([r_even, r_odd]))[0, 1]

    # Accuracy = average correlation with each half
    accuracy_even = torch.corrcoef(torch.stack([f_pred, r_even]))[0, 1]
    accuracy_odd = torch.corrcoef(torch.stack([f_pred, r_odd]))[0, 1]
    mean_accuracy = 0.5 * (accuracy_even + accuracy_odd)

    # Adjusted R² = mean_accuracy² / reliability
    adjusted_r2 = mean_accuracy ** 2 / reliability

    return adjusted_r2.item()
