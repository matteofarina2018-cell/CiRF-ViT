"""
Eigenspace Projection Utilities for vargp_direct

This module implements eigenspace projection for dimensionality reduction
of variational parameters. In eigenspace:
- K_tilde_b becomes DIAGONAL (eigenvalues on diagonal)
- K_tilde_inv_b is trivially computed as 1/eigenvalues
- m_b and V_b are reduced from M to n_b dimensions (~10-11 typically)

Reference: varGP utils.py:varGP() lines 5435-5444, 5619-5627
"""

import torch
from typing import Tuple, Optional

from ._constants import EIGVAL_TOL


def eigendecompose_K_tilde(
    K_tilde: torch.Tensor,
    eigval_tol: float = EIGVAL_TOL
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eigendecompose inducing kernel matrix K_tilde.

    Performs eigendecomposition of K_tilde and keeps only eigenvectors
    corresponding to eigenvalues above threshold. This reduces dimensionality
    from M (number of inducing points) to n_b.

    In eigenspace, K_tilde_b = diag(eigvals_b) is DIAGONAL, making
    inverse trivial: K_tilde_inv_b = diag(1/eigvals_b).

    Args:
        K_tilde: Inducing point kernel matrix, shape (M, M)
        eigval_tol: Eigenvalue threshold. Eigenvalues <= max(max_eigval * tol, tol)
                   are discarded. Default: 1e-4.

    Returns:
        B: Eigenvector matrix for kept eigenvalues, shape (M, n_b)
        eigvals_b: Kept eigenvalues in ascending order, shape (n_b,)
        ikeep: Boolean mask of kept eigenvalue indices, shape (M,)

    Note:
        torch.linalg.eigh returns eigenvalues in ASCENDING order.
        Eigenvectors are columns of the returned matrix.
    """
    # Eigendecomposition (symmetric, use lower triangular)
    eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')

    # Threshold: keep eigenvalues > max(max_eigval * tol, tol)
    threshold = max(eigvals.max().item() * eigval_tol, eigval_tol)
    ikeep = eigvals > threshold

    # Keep only large eigenvalues and corresponding eigenvectors
    B = eigvecs[:, ikeep]  # (M, n_b)
    eigvals_b = eigvals[ikeep]  # (n_b,)

    return B, eigvals_b, ikeep


def project_to_eigenspace(
    B: torch.Tensor,
    m: Optional[torch.Tensor] = None,
    V: Optional[torch.Tensor] = None,
    K: Optional[torch.Tensor] = None
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Project variational parameters and kernels into eigenspace.

    Given eigenvector matrix B, projects:
        m_b = B.T @ m           # Variational mean
        V_b = B.T @ V @ B       # Variational covariance (NOT diagonal!)
        K_b = K @ B             # Cross-kernel from training to eigenspace

    Args:
        B: Eigenvector matrix, shape (M, n_b)
        m: Variational mean, shape (M,). Optional.
        V: Variational covariance, shape (M, M). Optional.
        K: Cross-kernel matrix, shape (N, M). Optional.

    Returns:
        m_b: Projected mean, shape (n_b,), or None if m not provided
        V_b: Projected covariance, shape (n_b, n_b), or None if V not provided
        K_b: Projected cross-kernel, shape (N, n_b), or None if K not provided

    Note:
        V_b is generally NOT diagonal even though K_tilde_b is diagonal.
    """
    m_b = B.T @ m if m is not None else None
    V_b = B.T @ V @ B if V is not None else None
    K_b = K @ B if K is not None else None

    return m_b, V_b, K_b


def reproject_variational_params(
    B_old: torch.Tensor,
    B_new: torch.Tensor,
    m_b: torch.Tensor,
    V_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproject variational parameters when eigenspace changes after M-step.

    When M-step modifies kernel hyperparameters, K_tilde changes, so the
    eigenspace (B) changes. We need to transform m_b, V_b to the new eigenspace:

        V_b_new = B_new.T @ (B_old @ V_b_old @ B_old.T) @ B_new
        m_b_new = B_new.T @ B_old @ m_b_old

    Reference: varGP utils.py lines 5619-5627

    Args:
        B_old: Old eigenvector matrix, shape (M, n_b_old)
        B_new: New eigenvector matrix, shape (M, n_b_new)
        m_b: Variational mean in old eigenspace, shape (n_b_old,)
        V_b: Variational covariance in old eigenspace, shape (n_b_old, n_b_old)

    Returns:
        m_b_new: Reprojected mean, shape (n_b_new,)
        V_b_new: Reprojected covariance, shape (n_b_new, n_b_new)

    Warning:
        If eigenspace dimension increases, the reprojected V_b may have small
        eigenvalues due to the new dimensions having no prior information.
        This is handled by the E-step update which ensures V stays positive definite.
    """
    # Expand V_b back to full M-dimensional space, then project to new eigenspace
    # V = B_old @ V_b @ B_old.T
    # V_b_new = B_new.T @ V @ B_new
    V_b_new = B_new.T @ (B_old @ V_b @ B_old.T) @ B_new

    # Project m to new eigenspace
    # m = B_old @ m_b
    # m_b_new = B_new.T @ m
    m_b_new = B_new.T @ B_old @ m_b

    return m_b_new, V_b_new


def compute_KKtilde_inv_b(
    K_b: torch.Tensor,
    eigvals_b: torch.Tensor
) -> torch.Tensor:
    """Compute K @ K_tilde_inv efficiently in eigenspace.

    In eigenspace, K_tilde_inv_b = diag(1/eigvals_b) is DIAGONAL.
    So K @ K_tilde_inv = K_b @ K_tilde_inv_b is:
        KKtilde_inv_b[i,j] = K_b[i,j] / eigvals_b[j]

    This is a simple element-wise division, NOT a matrix solve!

    This quantity is used extensively in:
    - E-step: Computing g and G
    - Lambda moments: Computing posterior mean/variance

    Args:
        K_b: Cross-kernel in eigenspace, shape (N, n_b)
        eigvals_b: Eigenvalues of K_tilde (kept), shape (n_b,)

    Returns:
        KKtilde_inv_b: K @ K_tilde_inv in eigenspace, shape (N, n_b)

    Note:
        This is the 'a' variable in Matthew's code (varGP utils.py KKtilde_inv_b).
    """
    # Element-wise division: each column j divided by eigvals_b[j]
    return K_b / eigvals_b.unsqueeze(0)  # (N, n_b)


def compute_K_tilde_b_diagonal(eigvals_b: torch.Tensor) -> torch.Tensor:
    """Create diagonal K_tilde_b matrix from eigenvalues.

    In eigenspace, K_tilde_b is diagonal with eigenvalues on diagonal.

    Args:
        eigvals_b: Kept eigenvalues, shape (n_b,)

    Returns:
        K_tilde_b: Diagonal matrix, shape (n_b, n_b)
    """
    return torch.diag(eigvals_b)


def compute_K_tilde_inv_b_diagonal(eigvals_b: torch.Tensor) -> torch.Tensor:
    """Create diagonal K_tilde_inv_b matrix from eigenvalues.

    In eigenspace, K_tilde_inv_b is diagonal with 1/eigenvalues on diagonal.

    Args:
        eigvals_b: Kept eigenvalues, shape (n_b,)

    Returns:
        K_tilde_inv_b: Diagonal inverse matrix, shape (n_b, n_b)
    """
    return torch.diag(1.0 / eigvals_b)
