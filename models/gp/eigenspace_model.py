"""
Eigenspace-based Variational GP Model

This module contains the model class and state management for the eigenspace
(vargp_direct) training mode. Provides a clean interface using eigenspace
projection for computational efficiency.

Contents:
- DirectVariationalState: State container for eigenspace quantities
- DirectVGPModel: Main model class (owns kernel, likelihood, state)
- EigenspacePosterior: Posterior at query points (use model(X) to get)
- EigenspaceVariationalDistribution: Variational params interface

Usage:
    model = DirectVGPModel(kernel, likelihood, X_train, X_tilde, eigval_tol)

    # Get posterior at training points
    posterior = model(model.X_train)
    f_mean = model.likelihood.expected_firing_rate(posterior)

    # Access variational params
    model.variational_distribution.mean  # Full M-space
    model.state.m_b  # Eigenspace (for E-step)

    # After M-step changes kernel params, sync eigenspace
    model.recompute_eigenspace()
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch

from ._constants import EIGVAL_TOL, LAMBDA_VAR_CLAMP
from .eigenspace_utils import (
    eigendecompose_K_tilde,
    reproject_variational_params,
)


# ==============================================================================
# State Container
# ==============================================================================

@dataclass
class DirectVariationalState:
    """State container for direct variational GP.

    Stores all quantities needed for E-step, F-step, M-step in eigenspace.

    Attributes:
        m_b: Variational mean in eigenspace, shape (n_b,)
        V_b: Variational covariance in eigenspace, shape (n_b, n_b) - NOT diagonal!
        B: Eigenvector matrix, shape (M, n_b)
        eigvals_b: Kept eigenvalues, shape (n_b,)
        K_tilde_b: Inducing kernel in eigenspace - DIAGONAL, shape (n_b, n_b)
        K_b: Cross-kernel in eigenspace, shape (N, n_b)
        KKtilde_inv_b: K @ K_tilde_inv in eigenspace, shape (N, n_b)
        Kvec: Diagonal k(x_i, x_i), shape (N,)
        K_tilde: Full inducing kernel matrix, shape (M, M). Stored for efficient
            rank-1 column append during active learning (avoids O(M^2) kernel
            recomputation when extending model with one new inducing point).
        mask: Pixel mask from kernel (if use_mask=True), shape (n_pixels,) or None
    """
    m_b: torch.Tensor
    V_b: torch.Tensor
    B: torch.Tensor
    eigvals_b: torch.Tensor
    K_tilde_b: torch.Tensor
    K_b: torch.Tensor
    KKtilde_inv_b: torch.Tensor
    Kvec: torch.Tensor
    K_tilde: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


# ==============================================================================
# Eigenspace Computation (Shared Logic)
# ==============================================================================

def _compute_eigenspace_quantities(
    kernel,
    X_train: torch.Tensor,
    X_tilde: torch.Tensor,
    eigval_tol: float = EIGVAL_TOL
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute all eigenspace quantities from kernel and data.

    This is the core computation shared by:
    - _compute_initial_eigenspace() (model initialization)
    - _recompute_eigenspace() (after M-step changes kernel params)

    The eigenspace projection reduces dimensionality from M (inducing points)
    to n_b (kept eigenvalues), making K_tilde_b diagonal and K_tilde_inv trivial.

    Args:
        kernel: ArcCosineKernel instance
        X_train: Training inputs, shape (N, n_features)
        X_tilde: Inducing points, shape (M, n_features)
        eigval_tol: Eigenvalue threshold for projection

    Returns:
        B: Eigenvector matrix, shape (M, n_b)
        eigvals_b: Kept eigenvalues, shape (n_b,)
        K_b: Cross-kernel in eigenspace, K @ B, shape (N, n_b)
        K_tilde_b: Inducing kernel in eigenspace (DIAGONAL), shape (n_b, n_b)
        K_times_Ktilde_inv_b: K @ K_tilde_inv in eigenspace, shape (N, n_b)
        Kvec: Diagonal k(x_i, x_i), shape (N,)
        K_tilde: Full inducing kernel matrix, shape (M, M)
        mask: Pixel mask from kernel, or None
    """
    # -------------------------------------------------------------------------
    # Step 1: Compute kernel matrices using GPyTorch kernel
    # -------------------------------------------------------------------------
    with torch.no_grad():
        K_tilde = kernel(X_tilde, X_tilde).to_dense()  # (M, M) inducing kernel
        K = kernel(X_train, X_tilde).to_dense()        # (N, M) cross-kernel
        Kvec = kernel(X_train, diag=True)              # (N,) diagonal k(x_i, x_i)

    # Get mask if kernel uses masking
    mask = kernel._cached_mask if hasattr(kernel, '_cached_mask') else None

    # -------------------------------------------------------------------------
    # Step 2: Eigendecomposition of K_tilde
    # K_tilde = B @ diag(eigvals) @ B.T
    # We keep only eigenvectors with eigenvalues > threshold
    # -------------------------------------------------------------------------
    B, eigvals_b, _ = eigendecompose_K_tilde(K_tilde, eigval_tol)

    # -------------------------------------------------------------------------
    # Step 3: Project cross-kernel K to eigenspace
    # K_b = K @ B, shape (N, n_b)
    # -------------------------------------------------------------------------
    K_b = K @ B

    # -------------------------------------------------------------------------
    # Step 4: K_tilde in eigenspace is DIAGONAL
    # K_tilde_b = B.T @ K_tilde @ B = diag(eigvals_b)
    # -------------------------------------------------------------------------
    K_tilde_b = torch.diag(eigvals_b)

    # -------------------------------------------------------------------------
    # Step 5: Compute K @ K_tilde_inv in eigenspace
    # Since K_tilde_b is diagonal, K_tilde_inv_b = diag(1/eigvals_b)
    # So K @ K_tilde_inv = K_b @ diag(1/eigvals_b) = K_b / eigvals_b (element-wise)
    # This avoids expensive matrix solve - just element-wise division
    # -------------------------------------------------------------------------
    K_times_Ktilde_inv_b = K_b / eigvals_b.unsqueeze(0)  # (N, n_b)

    return B, eigvals_b, K_b, K_tilde_b, K_times_Ktilde_inv_b, Kvec, K_tilde, mask


# ==============================================================================
# State Initialization and Reprojection
# ==============================================================================

def _compute_initial_eigenspace(
    kernel,
    X_train: torch.Tensor,
    X_tilde: torch.Tensor,
    eigval_tol: float = EIGVAL_TOL
) -> DirectVariationalState:
    """Compute initial eigenspace state for model initialization.

    Creates DirectVariationalState with all quantities needed for training.
    Initializes variational parameters at the prior: m_b = 0, V_b = K_tilde_b.

    Args:
        kernel: ArcCosineKernel instance
        X_train: Training inputs, shape (N, n_features)
        X_tilde: Inducing points, shape (M, n_features)
        eigval_tol: Eigenvalue threshold for projection

    Returns:
        DirectVariationalState with initialized quantities
    """
    # Compute all eigenspace quantities (shared with _recompute_eigenspace)
    B, eigvals_b, K_b, K_tilde_b, K_times_Ktilde_inv_b, Kvec, K_tilde, mask = \
        _compute_eigenspace_quantities(kernel, X_train, X_tilde, eigval_tol)

    # Initialize variational parameters at prior
    n_b = len(eigvals_b)
    m_b = torch.zeros(n_b, dtype=X_train.dtype, device=X_train.device)  # Prior mean = 0
    V_b = K_tilde_b.clone()  # Prior covariance = K_tilde

    return DirectVariationalState(
        m_b=m_b,
        V_b=V_b,
        B=B,
        eigvals_b=eigvals_b,
        K_tilde_b=K_tilde_b,
        K_b=K_b,
        KKtilde_inv_b=K_times_Ktilde_inv_b,
        Kvec=Kvec,
        K_tilde=K_tilde,
        mask=mask,
    )


def _recompute_eigenspace(
    kernel,
    X_train: torch.Tensor,
    X_tilde: torch.Tensor,
    state: DirectVariationalState,
    eigval_tol: float = EIGVAL_TOL
) -> DirectVariationalState:
    """Recompute eigenspace after kernel hyperparameters change.

    When M-step modifies kernel hyperparameters, K_tilde changes, so the
    eigenspace (B, eigvals_b) changes. This function:
    1. Recomputes all eigenspace quantities with new kernel params
    2. Reprojects m_b, V_b from old eigenspace to new eigenspace

    Args:
        kernel: ArcCosineKernel instance (with updated hyperparameters)
        X_train: Training inputs, shape (N, n_features)
        X_tilde: Inducing points, shape (M, n_features)
        state: Current state with old eigenspace
        eigval_tol: Eigenvalue threshold for projection

    Returns:
        New DirectVariationalState with reprojected quantities
    """
    # Save old eigenspace quantities for reprojection
    B_old = state.B
    m_b_old = state.m_b
    V_b_old = state.V_b

    # Compute all eigenspace quantities with NEW kernel params
    B_new, eigvals_b, K_b, K_tilde_b, K_times_Ktilde_inv_b, Kvec, K_tilde, mask = \
        _compute_eigenspace_quantities(kernel, X_train, X_tilde, eigval_tol)

    # Reproject variational parameters from old eigenspace to new eigenspace
    # m_b_new = B_new.T @ B_old @ m_b_old
    # V_b_new = B_new.T @ (B_old @ V_b_old @ B_old.T) @ B_new
    m_b_new, V_b_new = reproject_variational_params(B_old, B_new, m_b_old, V_b_old)

    return DirectVariationalState(
        m_b=m_b_new,
        V_b=V_b_new,
        B=B_new,
        eigvals_b=eigvals_b,
        K_tilde_b=K_tilde_b,
        K_b=K_b,
        KKtilde_inv_b=K_times_Ktilde_inv_b,
        Kvec=Kvec,
        K_tilde=K_tilde,
        mask=mask,
    )


# ==============================================================================
# Posterior Moment Computation
# ==============================================================================

def _lambda_moments_eigenspace(state: DirectVariationalState, lambda_var_clamp: float = LAMBDA_VAR_CLAMP) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GP posterior moments using eigenspace quantities (internal).

    Used internally by EigenspacePosterior._compute_moments() for training data.
    External code should use model(X_train) to get posterior moments.

    Computes:
        lambda_m = K @ K_tilde_inv @ m = KKtilde_inv_b @ m_b
        lambda_var = Kvec + diag(a @ (V_b - K_tilde_b) @ a.T)
                   = Kvec + sum(a * (a @ (V_b - K_tilde_b)), dim=1)

    where a = KKtilde_inv_b.

    Reference: varGP utils.py:lambda_moments()

    Args:
        state: DirectVariationalState with current quantities

    Returns:
        lambda_m: Posterior mean at training points, shape (N,)
        lambda_var: Posterior variance at training points, shape (N,)
    """
    a = state.KKtilde_inv_b  # (N, n_b)

    # Mean: lambda_m = a @ m_b
    # TODO (cleanup): first GPU matmul in Phase 2 triggers a cuBLAS UserWarning ("no current CUDA
    # context, attempting to set primary context") due to thread context transition after the Phase 1
    # fit thread exits. Harmless — context is recovered automatically. Fix: warm up CUDA in the main
    # thread with a dummy op after the fit thread joins, before entering Phase 2.
    lambda_m = a @ state.m_b  # (N,)

    # Variance: lambda_var = Kvec + diag(a @ (V - K) @ a.T)
    V_minus_K = state.V_b - state.K_tilde_b  # (n_b, n_b)
    aV = a @ V_minus_K  # (N, n_b)
    lambda_var = state.Kvec + (a * aV).sum(dim=1)  # (N,)

    # Clamp for numerical stability
    lambda_var = torch.clamp(lambda_var, min=lambda_var_clamp)

    return lambda_m, lambda_var


# ==============================================================================
# GPyTorch-like Model Classes
# ==============================================================================

class EigenspacePosterior:
    """Posterior distribution at query points. Returned by DirectVGPModel(X).

    Provides GPyTorch-like .mean and .variance properties.

    This class computes posterior moments on construction using the same
    formulas as lambda_moments_eigenspace() and predict_eigenspace().
    """

    def __init__(
        self,
        kernel,
        state: DirectVariationalState,
        X_query: torch.Tensor,
        X_tilde: torch.Tensor,
        is_training_data: bool = False,
        lambda_var_clamp: float = LAMBDA_VAR_CLAMP
    ):
        """
        Args:
            kernel: ArcCosineKernel instance
            state: DirectVariationalState with eigenspace params
            X_query: (N_query, n_features) - points to evaluate posterior at
            X_tilde: (M, n_features) - inducing points
            is_training_data: If True, use precomputed KKtilde_inv_b from state
                (avoids recomputing kernel matrices for training points)
            lambda_var_clamp: Minimum posterior variance clamp
        """
        self._kernel = kernel
        self._state = state
        self._X_query = X_query
        self._X_tilde = X_tilde
        self._is_training_data = is_training_data
        self._lambda_var_clamp = lambda_var_clamp

        # Compute moments ONCE on construction
        self._mean, self._variance = self._compute_moments()

    def _compute_moments(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior moments at query points.

        For training points (is_training_data=True): uses precomputed
        state.KKtilde_inv_b for efficiency and exact equivalence with
        lambda_moments_eigenspace().

        For test points: computes cross-kernel and projects to eigenspace,
        matching predict_eigenspace() exactly.
        """
        if self._is_training_data:
            # Use precomputed KKtilde_inv_b for efficiency
            return _lambda_moments_eigenspace(self._state, self._lambda_var_clamp)

        # Test points: compute fresh
        with torch.no_grad():
            # Cross-kernel to inducing points
            K_query = self._kernel(self._X_query, self._X_tilde).to_dense()  # (N_query, M)
            Kvec_query = self._kernel(self._X_query, diag=True)  # (N_query,)

            # Project to eigenspace
            K_query_b = K_query @ self._state.B  # (N_query, n_b)

            # a = K @ K_tilde_inv (element-wise because K_tilde_b is diagonal)
            a = K_query_b / self._state.eigvals_b.unsqueeze(0)  # (N_query, n_b)

            # Posterior mean
            lambda_m = a @ self._state.m_b  # (N_query,)

            # Posterior variance
            V_minus_K = self._state.V_b - self._state.K_tilde_b
            aV = a @ V_minus_K  # (N_query, n_b)
            lambda_var = Kvec_query + (a * aV).sum(dim=1)  # (N_query,)
            lambda_var = torch.clamp(lambda_var, min=self._lambda_var_clamp)  # Numerical stability

        return lambda_m, lambda_var

    @property
    def mean(self) -> torch.Tensor:
        """Posterior mean lambda_m at query points. Shape (N_query,)."""
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        """Posterior variance lambda_var at query points. Shape (N_query,)."""
        return self._variance


class EigenspaceVariationalDistribution:
    """Variational distribution q(u) interface.

    Provides access to variational parameters in both full M-space
    and reduced n_b eigenspace.
    """

    def __init__(self, state: DirectVariationalState):
        self._state = state

    @property
    def mean(self) -> torch.Tensor:
        """Variational mean in full M-dimensional space: B @ m_b.

        Shape (M,).
        """
        return self._state.B @ self._state.m_b

    @property
    def mean_eigenspace(self) -> torch.Tensor:
        """Variational mean in eigenspace (direct access).

        Shape (n_b,). Returns same object as state.m_b.
        """
        return self._state.m_b

    @property
    def covariance_eigenspace(self) -> torch.Tensor:
        """Variational covariance in eigenspace (direct access).

        Shape (n_b, n_b). NOT diagonal!
        """
        return self._state.V_b

    @property
    def covariance(self) -> torch.Tensor:
        """Variational covariance in full M-dimensional space: B @ V_b @ B.T.

        Shape (M, M). WARNING: This is expensive for large M.
        """
        return self._state.B @ self._state.V_b @ self._state.B.T


class DirectVGPModel:
    """Eigenspace variational GP model for vargp_direct training mode.

    This class owns kernel, likelihood, training data, and variational state.
    It provides controlled mutation methods for state updates during training.

    Key methods:
    - model(X) returns EigenspacePosterior with .mean, .variance
    - model.update_variational_params(m_b, V_b) - update after E-step
    - model.recompute_eigenspace() - sync eigenspace after M-step

    Example usage:
        model = DirectVGPModel(kernel, likelihood, X_train, X_tilde, eigval_tol)

        # Get posterior at training points
        posterior = model(model.X_train)

        # Update variational params (E-step)
        model.update_variational_params(m_b_new, V_b_new)

        # After M-step changes kernel params, sync eigenspace
        model.recompute_eigenspace()
    """

    def __init__(
        self,
        kernel,
        likelihood,
        X_train: torch.Tensor,
        X_tilde: torch.Tensor,
        eigval_tol: float = EIGVAL_TOL,
        lambda_var_clamp: float = LAMBDA_VAR_CLAMP,
        _precomputed_state: Optional[DirectVariationalState] = None,
    ):
        """
        Args:
            kernel: ArcCosineKernel instance
            likelihood: PoissonLikelihood instance
            X_train: Training data, shape (N, n_features)
            X_tilde: Inducing points, shape (M, n_features)
            eigval_tol: Eigenvalue threshold for projection (default 1e-4)
            lambda_var_clamp: Minimum posterior variance clamp (default 1e-6)
            _precomputed_state: If provided, skip eigenspace computation and use
                this state directly. Used by extend_model_with_new_point to avoid
                redundant kernel evaluation when the state was already built
                efficiently via column append. Caller is responsible for ensuring
                consistency between the state and kernel/X_train/X_tilde.
        """
        self.kernel = kernel
        self.likelihood = likelihood
        self.X_train = X_train
        self.X_tilde = X_tilde
        self.eigval_tol = eigval_tol
        self.lambda_var_clamp = lambda_var_clamp

        if _precomputed_state is not None:
            self._state = _precomputed_state
        else:
            # Compute initial eigenspace
            self._state = _compute_initial_eigenspace(kernel, X_train, X_tilde, eigval_tol)

    def eval(self):
        """No-op for compatibility with code that calls model.eval()."""
        return self

    def train(self, mode=True):
        """No-op for compatibility with code that calls model.train()."""
        return self

    def update_variational_params(self, m_b: torch.Tensor, V_b: torch.Tensor) -> None:
        """Update variational parameters after E-step.

        Args:
            m_b: New variational mean in eigenspace, shape (n_b,)
            V_b: New variational covariance in eigenspace, shape (n_b, n_b)

        Note:
            V_b is symmetrized for numerical stability.
            Both tensors are detached from the autograd graph. m_b and V_b are
            updated by Newton's method (closed-form), not by gradient descent.
            Without detach, the E-step computation graph chains across iterations
            through A (which has requires_grad=True), causing unbounded memory
            growth: ~550 CUDA tensors per refit, reaching 19.9 GB at M=419.
        """
        self._state.m_b = m_b.detach()
        self._state.V_b = ((V_b + V_b.T) / 2).detach()  # Symmetrize for stability

    def recompute_eigenspace(self) -> None:
        """Recompute eigenspace after kernel hyperparameters change.

        Call this after M-step modifies kernel hyperparameters. Updates:
        - K_tilde, K, Kvec with current kernel hyperparameters
        - Eigendecomposition (B, eigvals_b)
        - Reprojected m_b, V_b in new eigenspace
        """
        self._state = _recompute_eigenspace(
            self.kernel, self.X_train, self.X_tilde, self._state, self.eigval_tol
        )

    @property
    def state(self) -> DirectVariationalState:
        """Access internal state for reading eigenspace quantities.

        Use model.update_variational_params() to update m_b, V_b.
        Use model.recompute_eigenspace() after M-step.
        """
        return self._state

    def __call__(self, X_query: torch.Tensor) -> EigenspacePosterior:
        """Compute posterior at query points.

        Args:
            X_query: (N_query, n_features) - points to evaluate

        Returns:
            EigenspacePosterior with .mean and .variance properties.

        Note:
            If X_query is self.X_train, uses precomputed KKtilde_inv_b.
        """
        # Check if this is the training data (same object reference)
        is_training_data = X_query is self.X_train

        return EigenspacePosterior(
            self.kernel, self._state, X_query, self.X_tilde, is_training_data,
            lambda_var_clamp=self.lambda_var_clamp
        )

    @property
    def variational_distribution(self) -> EigenspaceVariationalDistribution:
        """Access variational distribution q(u).

        Returns object with:
        - .mean: Full M-space mean (B @ m_b)
        - .mean_eigenspace: Reduced n_b-space mean (m_b)
        - .covariance_eigenspace: Reduced n_b-space covariance (V_b)
        """
        return EigenspaceVariationalDistribution(self._state)
