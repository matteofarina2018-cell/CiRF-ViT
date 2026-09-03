"""Small numerical helpers shared by the GP baseline's E/F/M steps.

Both functions come from the expected log-likelihood of the Poisson model with
an exponential link, in which the firing rate of a stimulus is

    f = exp(A * lambda + lambda0)

with `lambda` the latent GP value, `A` a gain and `lambda0` an offset. Because
the posterior over `lambda` is Gaussian with mean `lambda_m` and variance
`lambda_var`, both expectations below are available in closed form through the
log-normal moment exp(mu + sigma^2 / 2).
"""

import torch


def lambda0_given_A(
    A: torch.Tensor,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor
) -> torch.Tensor:
    """Closed-form optimal lambda0 given A.

    Derived from setting dL/d(lambda0) = 0 where L is the expected log-likelihood.
    This matches utils.py:lambda0_given_logA() but takes A directly (not logA).

    The expected log-likelihood contains:
        E[r*lambda0 - exp(A*lambda + lambda0)]
      = r*lambda0 - exp(lambda0)*E[exp(A*lambda)]
      = r*lambda0 - exp(lambda0)*exp(A*lambda_m + 0.5*A^2*lambda_var)

    Setting d/d(lambda0) = 0:
        sum(r) = exp(lambda0) * sum(exp(A*lambda_m + 0.5*A^2*lambda_var))

    Solution:
        lambda0 = log(sum(r)) - log(sum(exp(A*lambda_m + 0.5*A^2*lambda_var)))

    Args:
        A: Gain parameter (scalar tensor)
        r: Spike counts, shape (N,)
        lambda_m: GP posterior mean, shape (N,)
        lambda_var: GP posterior variance, shape (N,)

    Returns:
        Optimal lambda0 (scalar tensor)

    Raises:
        ValueError: If sum(r) <= 0 (no spikes in training data)
    """
    sumr = r.sum()

    # Guard against zero spike count which would cause log(0) = -inf
    if sumr <= 0:
        raise ValueError("All training spikes are zero. Data problem.")

    expexpr = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var)
    sumexpr = expexpr.sum()
    return torch.log(sumr) - torch.log(sumexpr)


def compute_f_mean(
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    A: torch.Tensor,
    lambda0: torch.Tensor
) -> torch.Tensor:
    """Compute expected firing rate.

    f_mean = exp(A * lambda_m + 0.5 * A^2 * lambda_var + lambda0)

    Args:
        lambda_m: Posterior mean, shape (N,)
        lambda_var: Posterior variance, shape (N,)
        A: Gain parameter (scalar)
        lambda0: Bias parameter (scalar)

    Returns:
        f_mean: Expected firing rate, shape (N,)
    """
    return torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)
