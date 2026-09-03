"""
Analytical Gradient Functions for Eigenspace M-Step

This module contains analytical gradient computation functions used by
mstep_eigenspace_analytical() for efficient M-step optimization.

These functions compute gradients of the negative ELBO (loss) with respect to
kernel hyperparameters without building the autograd graph.

DEFERRED REORGANIZATION:
------------------------
There are TWO gradient systems in the codebase:
1. Kernel-level gradients (analytical_gradients.py, analytical_gradients_vjp.py)
   - For --gradient-mode option
2. M-step gradients (this file)
   - For mstep_eigenspace_analytical()

These have different APIs but some shared formulas. Consolidation is deferred
to a future session to avoid breaking anything.

Key functions:
- compute_C_and_gradients: Compute C matrix and dC/d(hyperparameters)
- compute_kernel_and_gradients: Compute K and dK/d(hyperparameters)
- compute_lambda_moments_and_gradients: Compute moments and their gradients
- compute_loss_gradients: Compute dL/d(hyperparameters)

Reference: varGP utils.py:varGP() M-step closure
"""

import torch
from typing import Dict

from ._constants import LAMBDA_VAR_CLAMP


# ==============================================================================
# Analytical Gradient Functions for M-step Optimization
# ==============================================================================

def compute_C_and_gradients(kernel):
    """Compute C matrix and all dC/d(hyperparameters) matrices.

    Ports varGP utils.py:localker(grad=True) lines 3577-3631.

    The C matrix encodes receptive field (RF) structure:
        C = Amp * alpha[:, None] * C_smooth * alpha[None, :]

    where:
        alpha[i] = exp(-beta_factor * dist_center[i])
        C_smooth[i,j] = exp(-rho2_factor * dist_pairwise[i,j])
        beta_factor = exp(raw_m2log2beta)
        rho2_factor = exp(raw_mlog2rho2)

    Args:
        kernel: ArcCosineKernel instance with RF parameters

    Returns:
        C: Structured covariance matrix, shape (n_masked, n_masked)
        mask: Boolean mask for active pixels, shape (n_px,)
        dC: Dict of gradient matrices, each shape (n_masked, n_masked)
            Keys: 'Amp', 'eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2'

    Reference:
        varGP utils.py:localker() lines 3617-3627
    """
    # Get parameters
    Amp = kernel.Amp.squeeze()
    eps_0x = kernel.eps_0x.squeeze()
    eps_0y = kernel.eps_0y.squeeze()

    # Transform log-space parameters
    # beta_factor = exp(raw_m2log2beta) = 1/(4*beta^2)
    # rho2_factor = exp(raw_mlog2rho2) = 1/(2*rho^2)
    beta_factor = torch.exp(kernel.raw_m2log2beta.squeeze())
    rho2_factor = torch.exp(kernel.raw_mlog2rho2.squeeze())

    # Get pixel coordinates
    xcord = kernel.xcord.clone()
    ycord = kernel.ycord.clone()

    # Compute mask (with detached params for structural stability)
    mask = kernel.compute_mask()
    xcord = xcord[mask]
    ycord = ycord[mask]

    # Locality weights: distance from RF center
    dist_sq_center = (xcord - eps_0x)**2 + (ycord - eps_0y)**2
    logalpha = -beta_factor * dist_sq_center
    alpha = torch.exp(logalpha)

    # Smoothness kernel: pairwise pixel distances
    dx = xcord[:, None] - xcord[None, :]
    dy = ycord[:, None] - ycord[None, :]
    dist_sq_pairwise = dx**2 + dy**2
    logCsmooth = -rho2_factor * dist_sq_pairwise
    C_smooth = torch.exp(logCsmooth)

    # Full C matrix
    C = Amp * alpha[:, None] * C_smooth * alpha[None, :]

    # Symmetrize for numerical stability
    C = (C + C.T) / 2

    # Compute gradients (exact formulas from varGP utils.py lines 3619-3626)
    dC = {}

    # dC/d(Amp) = C / Amp
    dC['Amp'] = C / Amp

    # dC/d(eps_0x) = 2 * beta_factor * C * (xi + xj - 2*eps_0x)
    dC['eps_0x'] = 2 * beta_factor * C * (xcord[:, None] + xcord[None, :] - 2 * eps_0x)

    # dC/d(eps_0y) = 2 * beta_factor * C * (yi + yj - 2*eps_0y)
    dC['eps_0y'] = 2 * beta_factor * C * (ycord[:, None] + ycord[None, :] - 2 * eps_0y)

    # dC/d(raw_m2log2beta) = C * (logalpha_i + logalpha_j)
    dC['raw_m2log2beta'] = C * (logalpha[:, None] + logalpha[None, :])

    # dC/d(raw_mlog2rho2) = C * logCsmooth
    dC['raw_mlog2rho2'] = C * logCsmooth

    return C, mask, dC


def compute_kernel_and_gradients(x1, x2, C, dC, sigma_0, diag=False):
    """Compute kernel matrix K and all dK/d(hyperparameters) matrices.

    Ports varGP utils.py:acosker(dC=...) lines 3663-3813.

    For full matrix case (diag=False):
        K[i,j] = M[i,j] * J(theta[i,j])
        where M = sqrt(V1 * V2), V1 = x1.T C x1 + sigma_0^2, etc.

    For diagonal case (diag=True):
        K[i] = x1[i].T @ C @ x1[i] + sigma_0^2

    Args:
        x1: First input, shape (n1, n_features)
        x2: Second input, shape (n2, n_features) or None for diagonal
        C: Structured covariance matrix, shape (n_features, n_features)
        dC: Dict of dC matrices from compute_C_and_gradients()
        sigma_0: Kernel bias variance (scalar tensor)
        diag: If True, return only diagonal elements

    Returns:
        K: Kernel matrix, shape (n1, n2) or (n1,) if diag=True
        dK: Dict of gradient matrices, same shape as K
            Keys: 'sigma_0', 'Amp', 'eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2'

    Reference:
        varGP utils.py:acosker() lines 3699-3813
    """
    sigma_0_sq = sigma_0 ** 2
    n1 = x1.shape[0]

    # Compute quadratic forms x1.T C x1
    # CX1[i,:] = C @ x1[i,:], so CX1 has shape (n1, n_features)
    CX1 = x1 @ C  # (n1, n_features)
    V1 = (CX1 * x1).sum(dim=-1) + sigma_0_sq  # (n1,)

    if diag:
        # Diagonal case: K[i] = V1[i] = x1[i].T @ C @ x1[i] + sigma_0^2
        K = V1

        # Gradients
        dK = {}

        # dK/d(sigma_0) = 2 * sigma_0
        dK['sigma_0'] = torch.full_like(K, 2 * sigma_0.item())

        # dK/d(C-params): dK[key][i] = x1[i].T @ dC[key] @ x1[i]
        for key, dC_val in dC.items():
            if key == 'sigma_0':
                continue
            # dCX1[i,:] = dC[key] @ x1[i,:], so (dCX1 * x1).sum(-1) = diag(x1 @ dC @ x1.T)
            dCX1 = x1 @ dC_val  # (n1, n_features)
            dK[key] = (dCX1 * x1).sum(dim=-1)  # (n1,)

        return K, dK

    # Full matrix case
    n2 = x2.shape[0]

    CX2 = x2 @ C  # (n2, n_features)
    V2 = (CX2 * x2).sum(dim=-1) + sigma_0_sq  # (n2,)

    X1 = torch.sqrt(V1)  # (n1,)
    X2 = torch.sqrt(V2)  # (n2,)

    # M = sqrt(V1 * V2) = X1 * X2
    X1X2 = X1[:, None] * X2[None, :]  # (n1, n2)

    # Cross-term: x1.T C x2 + sigma_0^2
    x1x2 = (CX1 @ x2.T) + sigma_0_sq  # (n1, n2)

    # Normalized inner product
    eps = 1e-7
    cosdelta = torch.clamp(x1x2 / (X1X2 + eps), -1.0 + eps, 1.0 - eps)

    # Angle
    delta = torch.arccos(cosdelta)

    # Angular term: J(theta) = (sin(theta) + (pi - theta)cos(theta)) / pi
    sin_delta = torch.sqrt(torch.clamp(1.0 - cosdelta ** 2, min=eps))
    J = (sin_delta + (torch.pi - delta) * cosdelta) / torch.pi

    # Kernel: K = M * J
    K = X1X2 * J

    # ====== Gradients ======
    dK = {}

    # --- dK/d(sigma_0) ---
    # From varGP utils.py lines 3722-3730
    # dX1X2 = sigma_0^2 * (X2/X1 + X1/X2) in broadcasted form
    dX1X2_sigma0 = sigma_0_sq * (X2[None, :] / X1[:, None] + X1[:, None] / X2[None, :])
    dcosdelta_sigma0 = (2 * sigma_0_sq - cosdelta * dX1X2_sigma0) / X1X2
    dJ_sigma0 = -(delta - torch.pi) * dcosdelta_sigma0 / torch.pi
    dK['sigma_0'] = (X1X2 * dJ_sigma0 + dX1X2_sigma0 * J) / sigma_0

    # --- dK/d(C-params) ---
    # From varGP utils.py lines 3734-3747
    for key, dC_val in dC.items():
        if key == 'sigma_0':
            continue

        # dX1[i] = 0.5 * x1[i].T @ dC @ x1[i] / X1[i]
        dCX1 = x1 @ dC_val  # (n1, n_features)
        dX1 = 0.5 * (dCX1 * x1).sum(dim=-1) / X1  # (n1,)

        # dX2[j] = 0.5 * x2[j].T @ dC @ x2[j] / X2[j]
        dCX2 = x2 @ dC_val  # (n2, n_features)
        dX2 = 0.5 * (dCX2 * x2).sum(dim=-1) / X2  # (n2,)

        # dX1X2[i,j] = dX1[i] * X2[j] + X1[i] * dX2[j]
        dX1X2 = dX1[:, None] * X2[None, :] + X1[:, None] * dX2[None, :]

        # dcosdelta = (x1.T @ dC @ x2 - cosdelta * dX1X2) / X1X2
        dx1x2 = dCX1 @ x2.T  # (n1, n2)
        dcosdelta = (dx1x2 - cosdelta * dX1X2) / X1X2

        # dJ = -(delta - pi) * dcosdelta / pi
        dJ = -(delta - torch.pi) * dcosdelta / torch.pi

        # dK[key] = X1X2 * dJ + dX1X2 * J
        dK[key] = X1X2 * dJ + dX1X2 * J

    return K, dK


def compute_lambda_moments_and_gradients(
    K_b, K_tilde_b, Kvec, m_b, V_b,
    dK_b, dK_tilde_b, dKvec,
    K_tilde_inv_b,
    lambda_var_clamp: float = LAMBDA_VAR_CLAMP
):
    """Compute posterior moments and their gradients w.r.t. hyperparameters.

    Ports varGP utils.py:lambda_moments() lines 3937-3952.

    Moments:
        a = K_b @ K_tilde_inv_b  (projection vector)
        lambda_m = a @ m_b
        lambda_var = Kvec + diag(a @ (V_b - K_tilde_b) @ a.T)

    Gradients:
        da[key] = (dK_b[key] - a @ dK_tilde_b[key]) @ K_tilde_inv_b
        dlambda_m[key] = da[key] @ m_b
        dlambda_var[key] = dKvec[key]
                         + 2 * diag(da[key] @ V_b @ a.T)
                         - diag(dK_b[key] @ a.T)
                         - diag(K_b @ da[key].T)

    Args:
        K_b: Cross-kernel in eigenspace, shape (N, n_b)
        K_tilde_b: Inducing kernel in eigenspace, shape (n_b, n_b)
        Kvec: Diagonal self-kernel, shape (N,)
        m_b: Variational mean in eigenspace, shape (n_b,)
        V_b: Variational covariance in eigenspace, shape (n_b, n_b)
        dK_b: Dict of dK_b matrices, each shape (N, n_b)
        dK_tilde_b: Dict of dK_tilde_b matrices, each shape (n_b, n_b)
        dKvec: Dict of dKvec vectors, each shape (N,)
        K_tilde_inv_b: Inverse of K_tilde_b, shape (n_b, n_b)
            Can be diagonal (1D tensor of eigenvalues) or full matrix

    Returns:
        lambda_m: Posterior mean, shape (N,)
        lambda_var: Posterior variance, shape (N,)
        dlambda_m: Dict of mean gradients, each shape (N,)
        dlambda_var: Dict of variance gradients, each shape (N,)

    Reference:
        varGP utils.py:lambda_moments() lines 3942-3952
    """
    # Handle both diagonal (eigenvalues) and full matrix cases
    if K_tilde_inv_b.dim() == 1:
        # Diagonal case: K_tilde_inv_b is eigenvalues, use element-wise multiply
        a = K_b * K_tilde_inv_b[None, :]  # (N, n_b)
        is_diagonal = True
    else:
        # Full matrix case: use matrix multiply
        a = K_b @ K_tilde_inv_b  # (N, n_b)
        is_diagonal = False

    # Posterior mean
    lambda_m = a @ m_b  # (N,)

    # Posterior variance: lambda_var = Kvec + diag(a @ (V - K) @ a.T)
    V_minus_K = V_b - K_tilde_b  # (n_b, n_b)
    aV = a @ V_minus_K  # (N, n_b)
    lambda_var = Kvec + (a * aV).sum(dim=-1)  # (N,)

    # Clamp for numerical stability
    lambda_var = torch.clamp(lambda_var, min=lambda_var_clamp)

    # ====== Gradients ======
    dlambda_m = {}
    dlambda_var = {}

    for key in dK_b.keys():
        # da/dtheta = (dK/dtheta - a @ dK_tilde/dtheta) @ K_tilde_inv
        if is_diagonal:
            da_key = (dK_b[key] - a @ dK_tilde_b[key]) * K_tilde_inv_b[None, :]  # (N, n_b)
        else:
            da_key = (dK_b[key] - a @ dK_tilde_b[key]) @ K_tilde_inv_b  # (N, n_b)

        # dlambda_m[key] = da[key] @ m
        dlambda_m[key] = da_key @ m_b  # (N,)

        # dlambda_var[key] = dKvec[key]
        #                  + 2*diag(da @ V @ a.T)   = einsum('ij,ji->i', 2*da, V@a.T)
        #                  - diag(dK @ a.T)        = einsum('ij,ij->i', dK, a)
        #                  - diag(K @ da.T)        = einsum('ij,ij->i', K, da)

        # Term 1: dKvec[key]
        term1 = dKvec[key]

        # Term 2: 2 * diag(da @ V @ a.T)
        Va_T = V_b @ a.T  # (n_b, N)
        term2 = 2 * torch.einsum('ij,ji->i', da_key, Va_T)  # (N,)

        # Term 3: -diag(dK @ a.T) = -sum(dK * a, dim=1)
        term3 = -torch.einsum('ij,ij->i', dK_b[key], a)  # (N,)

        # Term 4: -diag(K @ da.T) = -sum(K * da, dim=1)
        term4 = -torch.einsum('ij,ij->i', K_b, da_key)  # (N,)

        dlambda_var[key] = term1 + term2 + term3 + term4  # (N,)

    return lambda_m, lambda_var, dlambda_m, dlambda_var


def compute_loss_gradients(
    r, f_mean, A, m_b, V_b, K_tilde_b, K_tilde_inv_b,
    dlambda_m, dlambda_var, dK_tilde_b
):
    """Compute gradients of negative ELBO w.r.t. kernel hyperparameters.

    Loss = -loglikelihood + KL
    dL/dtheta = -dloglikelihood/dtheta + dKL/dtheta

    Ports varGP utils.py:compute_loglikelihood() lines 4111-4117 and
    varGP utils.py:compute_KL_div() lines 4143-4150.

    Args:
        r: Spike counts, shape (N,)
        f_mean: Expected firing rate, shape (N,)
        A: Gain parameter (scalar)
        m_b: Variational mean in eigenspace, shape (n_b,)
        V_b: Variational covariance in eigenspace, shape (n_b, n_b)
        K_tilde_b: Inducing kernel in eigenspace, shape (n_b, n_b)
        K_tilde_inv_b: Inverse of K_tilde_b - can be 1D (diagonal eigenvalues)
            or 2D (full matrix from solve())
        dlambda_m: Dict of mean gradients from compute_lambda_moments_and_gradients()
        dlambda_var: Dict of variance gradients from compute_lambda_moments_and_gradients()
        dK_tilde_b: Dict of dK_tilde_b matrices, each shape (n_b, n_b)

    Returns:
        dL: Dict of loss gradients w.r.t. each hyperparameter (scalars)

    Reference:
        varGP utils.py:compute_loglikelihood() lines 4111-4117
        varGP utils.py:compute_KL_div() lines 4143-4150
    """
    # Check if K_tilde_inv_b is diagonal (eigenvalues) or full matrix
    is_diagonal = K_tilde_inv_b.dim() == 1

    # ====== dloglikelihood/dtheta ======
    # From varGP utils.py line 4116:
    # dloglikelihood[key] = A*r@dlambda_m[key] - A*f_mean@dlambda_m[key] - 0.5*A^2*f_mean@dlambda_var[key]

    dloglikelihood = {}
    for key in dlambda_m.keys():
        dloglikelihood[key] = (
            A * (r @ dlambda_m[key])
            - A * (f_mean @ dlambda_m[key])
            - 0.5 * A * A * (f_mean @ dlambda_var[key])
        )

    # ====== dKL/dtheta ======
    # From varGP utils.py lines 4133-4148:
    # c = V @ K_tilde_inv
    # b = K_tilde_inv @ m
    # B = dK_tilde[key] @ K_tilde_inv
    # dKL[key] = 0.5*trace(B) - 0.5*trace(c@B) - 0.5*b.T@(B@m)

    if is_diagonal:
        # Diagonal case: K_tilde_inv_b is eigenvalues
        c = V_b * K_tilde_inv_b[None, :]  # (n_b, n_b) - scales columns
        b = K_tilde_inv_b * m_b  # (n_b,)
    else:
        # Full matrix case
        c = V_b @ K_tilde_inv_b  # (n_b, n_b)
        b = K_tilde_inv_b @ m_b  # (n_b,)

    dKL = {}
    for key in dK_tilde_b.keys():
        if is_diagonal:
            # B = dK_tilde_b[key] @ K_tilde_inv_b (diagonal on right)
            B = dK_tilde_b[key] * K_tilde_inv_b[None, :]  # (n_b, n_b)
        else:
            # Full matrix case
            B = dK_tilde_b[key] @ K_tilde_inv_b  # (n_b, n_b)

        # Term 1: 0.5 * trace(B)
        term1 = 0.5 * torch.trace(B)

        # Term 2: -0.5 * trace(c @ B)
        term2 = -0.5 * torch.trace(c @ B)

        # Term 3: -0.5 * b.T @ (B @ m)
        term3 = -0.5 * (b @ (B @ m_b))

        dKL[key] = term1 + term2 + term3

    # ====== Final gradient: dL = -dloglikelihood + dKL ======
    dL = {}
    for key in dlambda_m.keys():
        dL[key] = -dloglikelihood[key] + dKL[key]

    return dL
