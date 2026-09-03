"""
Poisson Likelihood for GPyTorch

This module implements the Poisson likelihood with exponential link function
for modeling neural spike count data.

Model: r ~ Poisson(f(x)) where f(x) = exp(A·λ(x) + λ₀)
- λ(x) is the latent GP function
- A is the gain parameter (learnable)
- λ₀ is the bias parameter (baseline log-firing rate, learnable)
"""

import warnings

import torch
import gpytorch
from gpytorch.likelihoods import Likelihood
from gpytorch.constraints import Positive


class PoissonLikelihood(Likelihood):
    """Poisson likelihood with exponential link function.

    Model: r ~ Poisson(exp(A·λ + λ₀))

    For variational inference with q(λ) = N(μ, σ²), the expected log-likelihood is:
        E[log p(r|λ)] = E[r·(A·λ + λ₀) - exp(A·λ + λ₀) - log(r!)]
                      = r·(A·μ + λ₀) - exp(A·μ + A²σ²/2 + λ₀) - log(r!)

    We omit log(r!) since it's constant w.r.t. parameters.

    Parameters
    ----------
    A_init : float
        Initial value for gain parameter A (default: 1.0)
    lambda0_init : float
        Initial value for bias parameter λ₀ (default: 0.0)

    Attributes
    ----------
    raw_A : Parameter
        Unconstrained parameter (logA), where A = exp(raw_A)
    A : Property
        Constrained (positive) A value
    lambda0 : Parameter
        Bias parameter (unconstrained)
    """

    # Bounds for likelihood parameters (generous — only catch absurd LBFGS overshoots)
    A_MAX = 10.0         # Typical A ~ 0.01; A > 10 is extreme
    LAMBDA0_MIN = -50.0
    LAMBDA0_MAX = 50.0

    def __init__(self, A_init=1.0, lambda0_init=0.0):
        super().__init__()

        # Register A parameter with positivity constraint
        self.register_parameter(
            name='raw_A',
            parameter=torch.nn.Parameter(torch.zeros(1))
        )

        # Use exp/log transform (A = exp(raw_A), matching varGP's logA)
        self.register_constraint('raw_A', Positive(transform=torch.exp, inv_transform=torch.log))

        self.A = A_init  # Set via property to apply inverse transform

        # Register lambda0 parameter (unconstrained)
        self.register_parameter(
            name='lambda0',
            parameter=torch.nn.Parameter(torch.tensor([lambda0_init]))
        )

    @property
    def A(self):
        """Get the constrained A value (positive)."""
        return self.raw_A_constraint.transform(self.raw_A)

    @A.setter
    def A(self, value):
        """Set A via inverse transform."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float)
        self.initialize(raw_A=self.raw_A_constraint.inverse_transform(value))

    def params_in_bounds(self):
        """Check whether all likelihood parameters are within valid bounds.

        READ-ONLY check. Call at the top of LBFGS closures to reject trial
        steps before expensive computation.

        Bounds: A in (0, A_MAX], lambda0 in [LAMBDA0_MIN, LAMBDA0_MAX].

        Returns
        -------
        bool
            True if all parameters are in bounds.
        """
        with torch.no_grad():
            a_val = self.A.item()
            if a_val <= 0 or a_val > self.A_MAX:
                return False
            l0_val = self.lambda0.item()
            if l0_val < self.LAMBDA0_MIN or l0_val > self.LAMBDA0_MAX:
                return False
        return True

    def clamp_params(self):
        """Clamp likelihood parameters to valid bounds (projected gradient descent).

        Call after optimizer.step() to enforce parameter bounds.
        Emits a warning listing which parameters were out of bounds,
        since this should not happen if the LBFGS bounds guard is working.

        Bounds: A in (0, A_MAX], lambda0 in [LAMBDA0_MIN, LAMBDA0_MAX].
        """
        with torch.no_grad():
            violated = []

            max_raw_A = self.raw_A_constraint.inverse_transform(
                torch.tensor(self.A_MAX, device=self.raw_A.device, dtype=self.raw_A.dtype)
            )
            if self.raw_A.item() > max_raw_A.item():
                violated.append(f"A={self.A.item():.4g} > {self.A_MAX}")
            self.raw_A.clamp_(max=max_raw_A.item())

            l0_val = self.lambda0.item()
            if l0_val < self.LAMBDA0_MIN or l0_val > self.LAMBDA0_MAX:
                violated.append(f"lambda0={l0_val:.4g} outside [{self.LAMBDA0_MIN}, {self.LAMBDA0_MAX}]")
            self.lambda0.clamp_(self.LAMBDA0_MIN, self.LAMBDA0_MAX)

            if violated:
                warnings.warn(
                    f"clamp_params: parameters escaped bounds — {', '.join(violated)}. "
                    f"This suggests the optimizer took a step the bounds guard did not catch."
                )

    def expected_log_prob(self, target, input):
        """Expected log probability under the variational distribution.

        Args:
            target: Observed spike counts, shape (n_samples,)
            input: GP output MultivariateNormal with mean and variance

        Returns:
            Expected log probability, summed over samples (scalar)
        """
        mu = input.mean        # (n_samples,)
        var = input.variance   # (n_samples,)

        A = self.A.squeeze()
        lambda0 = self.lambda0.squeeze()

        # E[r·(A·λ + λ₀) - exp(A·λ + λ₀)]
        # = r·(A·μ + λ₀) - exp(A·μ + A²σ²/2 + λ₀)
        log_prob = target * (A * mu + lambda0) - torch.exp(A * mu + 0.5 * A**2 * var + lambda0)

        return log_prob.sum(-1)

    def forward(self, function_samples):
        """Return Poisson distribution given function samples.

        Used for sampling from the likelihood.

        Args:
            function_samples: Samples of λ from the GP, shape (..., n_samples)

        Returns:
            Poisson distribution with rate = exp(A·λ + λ₀)
        """
        A = self.A.squeeze()
        lambda0 = self.lambda0.squeeze()
        rate = torch.exp(A * function_samples + lambda0)
        return torch.distributions.Poisson(rate=rate)

    def expected_firing_rate(self, posterior) -> torch.Tensor:
        """Compute expected firing rate from posterior moments.

        f_mean = exp(A * λ_m + 0.5 * A² * λ_var + λ₀)

        This is the expected value of exp(A*λ + λ₀) when λ ~ N(λ_m, λ_var).

        Args:
            posterior: Object with .mean and .variance attributes
                (e.g., EigenspacePosterior from DirectVGPModel)

        Returns:
            Expected firing rate, shape matching posterior.mean
        """
        A = self.A.squeeze()
        lambda0 = self.lambda0.squeeze()
        return torch.exp(A * posterior.mean + 0.5 * A * A * posterior.variance + lambda0)


def test_likelihood():
    """Basic test for PoissonLikelihood."""
    import numpy as np

    # Create likelihood
    likelihood = PoissonLikelihood(A_init=0.5, lambda0_init=1.0)

    print(f"A: {likelihood.A.item():.4f}")
    print(f"lambda0: {likelihood.lambda0.item():.4f}")

    # Create fake GP output
    mean = torch.tensor([0.0, 1.0, 2.0])
    var = torch.tensor([0.5, 0.5, 0.5])

    # Create MultivariateNormal-like object
    class FakeGPOutput:
        def __init__(self, mean, variance):
            self.mean = mean
            self.variance = variance

    gp_output = FakeGPOutput(mean, var)
    target = torch.tensor([1.0, 5.0, 10.0])

    # Compute expected log prob
    log_prob = likelihood.expected_log_prob(target, gp_output)
    print(f"Expected log prob: {log_prob.item():.4f}")

    # Verify manually
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    manual = (target * (A * mean + lambda0) - torch.exp(A * mean + 0.5 * A**2 * var + lambda0)).sum()
    print(f"Manual computation: {manual.item():.4f}")

    assert torch.allclose(log_prob, manual), "Mismatch!"
    print("PASS: PoissonLikelihood test passed!")

    # Test params_in_bounds
    lik_test = PoissonLikelihood(A_init=0.5, lambda0_init=1.0)
    assert lik_test.params_in_bounds(), "Fresh likelihood should be in bounds"
    with torch.no_grad():
        lik_test.lambda0.fill_(100.0)
    assert not lik_test.params_in_bounds(), "lambda0=100 should be out of bounds"
    with torch.no_grad():
        lik_test.lambda0.fill_(1.0)
    assert lik_test.params_in_bounds(), "Reset likelihood should be in bounds"
    print("PASS: params_in_bounds test passed!")

    # Test clamp_params
    lik_test2 = PoissonLikelihood(A_init=0.5, lambda0_init=1.0)
    with torch.no_grad():
        lik_test2.lambda0.fill_(100.0)
    import warnings as _w
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        lik_test2.clamp_params()
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert "escaped bounds" in str(w[0].message)
    assert lik_test2.lambda0.item() == PoissonLikelihood.LAMBDA0_MAX
    print("PASS: clamp_params test passed!")

    return True
