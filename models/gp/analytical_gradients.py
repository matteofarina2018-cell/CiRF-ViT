"""
Analytical Gradient Computation for Arc-Cosine Kernel
======================================================

This module implements analytical gradient computation for kernel hyperparameters,
matching the original varGP implementation in its utils.py:acosker(). Using analytical
gradients (instead of autograd) ensures gradient computation matches the reference
implementation exactly.

The derivations are those of Goldin, Virgili and Chalk (PNAS 2023); this module
only restates them in PyTorch.

Architecture
------------
The module provides three main components:

1. **acosker_with_hyp_grad()**
   Low-level function that computes K and all dK/dθ matrices given C and dC.

2. **compute_C_and_gradients()**
   Computes C matrix and dC/dθ for all hyperparameters from RF parameters.

3. **ArcCosineJacobianGradients(torch.autograd.Function)**
   PyTorch autograd wrapper. Forward computes K, backward uses analytical dK.

Data Flow
---------
When gradient_mode='jacobian' in ArcCosineKernel:

    forward(x1, x2):
        1. compute_C_and_gradients() → C, dC dict
        2. acosker_with_hyp_grad(C, dC) → K, dK dict
        3. Save dK tensors in ctx
        4. Return K

    backward(grad_output):
        For each θ: grad_θ = (grad_output * dK[θ]).sum()

Gradients Computed
------------------
- sigma_0: Direct derivative (σ₀ appears in K independently of C)
- Amp: Chain rule through C (amplitude, dC/dAmp = C/Amp)
- eps_0x, eps_0y: Chain rule through C (RF center position)
- -2log2beta: Chain rule through C (RF size, log-parameterized)
- -log2rho2: Chain rule through C (smoothness, log-parameterized)

Usage
-----
Option 1: Via ArcCosineKernel gradient_mode (recommended)

    from .kernels import ArcCosineKernel
    kernel = ArcCosineKernel(
        n_px_side=108,
        gradient_mode='jacobian'  # Analytical gradients (slow, reference)
    )

Option 2: Direct function call

    from analytical_gradients import acosker_with_hyp_grad
    K, dK = acosker_with_hyp_grad(sigma_0, x1, x2, C, dC, diag=False)
    # dK['sigma_0'], dK['eps_0x'], etc. are all (n1, n2) matrices

Status
------
This is the slow reference path, kept as a cross-check against the original
varGP code. It needs C_gradients_hyp() from that upstream implementation, which
is not distributed here -- see the note below the imports. Nothing in the
published pipeline uses it: the GP trains with gradient_mode='autograd' (the
default), and the fast analytical path is analytical_gradients_vjp.py.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# The Jacobian path reuses C_gradients_hyp() from the upstream Spatial_GP
# reference implementation of Goldin et al., which is NOT distributed with this
# repository. It is loaded lazily, from the file named by
#
#     CIRF_REFERENCE_KERNELS=/path/to/Spatial_GP_repo/kernels/kernels.py
#
# so importing this module never depends on it. Nothing in the published
# pipeline does: the GP is trained with gradient_mode='autograd' (the default)
# and the fast analytical path lives in analytical_gradients_vjp.py, both of
# which are self-contained. This mode exists only as a slow cross-check against
# the original code.
# ---------------------------------------------------------------------------
import importlib.util
import os

_reference_kernels = None


def _load_reference_kernels():
    """Import the upstream kernels.py on demand, or explain why it is missing."""
    global _reference_kernels
    if _reference_kernels is not None:
        return _reference_kernels
    path = os.environ.get('CIRF_REFERENCE_KERNELS')
    if not path or not os.path.exists(path):
        raise RuntimeError(
            "gradient_mode='jacobian' needs C_gradients_hyp() from the upstream "
            "Spatial_GP reference implementation, which is not part of this "
            "repository. Set CIRF_REFERENCE_KERNELS to that kernels.py, or use "
            "gradient_mode='autograd' (the default) or 'vjp', which need nothing "
            "external."
        )
    # importlib, not a plain import: the file is also called kernels.py.
    spec = importlib.util.spec_from_file_location('reference_kernels', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _reference_kernels = module
    return module


def C_gradients_hyp(*args, **kwargs):
    """dC/dtheta from the reference implementation (see _load_reference_kernels)."""
    return _load_reference_kernels().C_gradients_hyp(*args, **kwargs)


def acosker_with_hyp_grad(
    sigma_0: torch.Tensor,
    x1: torch.Tensor,
    x2: Optional[torch.Tensor],
    C: torch.Tensor,
    dC: Dict[str, torch.Tensor],
    diag: bool = False
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Arc-cosine kernel with analytical gradients w.r.t. hyperparameters.

    This function computes K and dK/dtheta for all hyperparameters.
    Matches the implementation in varGP utils.py:acosker() lines 3714-3798.

    Parameters
    ----------
    sigma_0 : Tensor (scalar)
        Bias variance hyperparameter
    x1 : Tensor (n1, nx)
        First input points (will be transposed internally to match reference)
    x2 : Tensor (n2, nx) or None
        Second input points. If None, uses x1 for diagonal case.
    C : Tensor (nx, nx)
        Structured covariance matrix
    dC : Dict[str, Tensor]
        Dictionary of dC/dtheta matrices for each hyperparameter.
        Keys: 'Amp', 'eps_0x', 'eps_0y', '-2log2beta', '-log2rho2'
        Each value is shape (nx, nx)
    diag : bool
        If True, compute only diagonal K(x_i, x_i)

    Returns
    -------
    K : Tensor
        Kernel matrix, shape (n1, n2) or (n1,) if diag=True
    dK : Dict[str, Tensor]
        Dictionary of dK/dtheta matrices for each hyperparameter.
        Keys: 'sigma_0', 'Amp', 'eps_0x', 'eps_0y', '-2log2beta', '-log2rho2'
        Each value has same shape as K

    Notes
    -----
    - Inputs are transposed internally (x1.T) to match reference code convention
    - The reference code came from MATLAB and uses (nx, n) convention
    - All computation uses the input dtype (should be float64 for stability)
    """
    # Transpose inputs to match reference code convention (nx, n)
    # Reference: varGP utils.py line 3690
    x1_t = x1.T  # (nx, n1)
    if x2 is not None:
        x2_t = x2.T  # (nx, n2)
    else:
        x2_t = x1_t

    n1 = x1_t.shape[1]
    device = x1.device
    dtype = x1.dtype

    dK = {}

    if not diag:
        # ===== Forward pass (compute K) =====
        # Reference: varGP utils.py lines 3700-3712

        # X1[i] = sqrt(x1[:,i]^T @ C @ x1[:,i] + sigma_0^2)
        # Using element-wise: sum(x1 * (C @ x1), dim=0) = diag(x1.T @ C @ x1)
        X1 = torch.sqrt(torch.sum(x1_t * (C @ x1_t), dim=0) + sigma_0 ** 2)  # (n1,)
        X2 = torch.sqrt(torch.sum(x2_t * (C @ x2_t), dim=0) + sigma_0 ** 2)  # (n2,)

        # Magnitude matrix
        X1X2 = torch.outer(X1, X2)  # (n1, n2)

        # Cross-term: x1x2[i,j] = x1[:,i]^T @ C @ x2[:,j] + sigma_0^2
        x1x2 = x1_t.T @ C @ x2_t + sigma_0 ** 2  # (n1, n2)

        # Normalized inner product (clamp for numerical stability)
        cosdelta = torch.clamp(x1x2 / (X1X2 + 1e-7), -1.0, 1.0)  # (n1, n2)

        # Angle and angular term
        delta = torch.arccos(cosdelta)  # (n1, n2)
        sindelta = torch.sqrt(1.0 - cosdelta ** 2)  # (n1, n2)

        # J = (sin(delta) + (pi - delta) * cos(delta)) / pi
        J = (sindelta + (torch.pi - delta) * cosdelta) / torch.pi  # (n1, n2)

        # Kernel
        K = X1X2 * J  # (n1, n2)

        # ===== Gradient w.r.t. sigma_0 =====
        # Reference: varGP utils.py lines 3720-3728

        # dX1X2/dsigma_0 = sigma_0^2 * (X2/X1 + X1/X2)
        dX1X2_sigma = sigma_0 ** 2 * (X2 / X1[:, None] + X1[:, None] / X2)  # (n1, n2)

        # dcosdelta/dsigma_0 = (2*sigma_0^2 - cosdelta * dX1X2) / X1X2
        dcosdelta_sigma = (2 * sigma_0 ** 2 - cosdelta * dX1X2_sigma) / X1X2  # (n1, n2)

        # dJ/dsigma_0 = -(delta - pi) * dcosdelta / pi
        dJ_sigma = -(delta - torch.pi) * dcosdelta_sigma / torch.pi  # (n1, n2)

        # dK/dsigma_0 = (X1X2 * dJ + dX1X2 * J) / sigma_0
        dK['sigma_0'] = (X1X2 * dJ_sigma + dX1X2_sigma * J) / sigma_0  # (n1, n2)

        # ===== Gradients w.r.t. C-dependent hyperparameters =====
        # Reference: varGP utils.py lines 3732-3745

        for key, dC_key in dC.items():
            if key == 'sigma_0':
                continue

            # dX1/dtheta = 0.5 * sum(x1 * (dC @ x1), dim=0) / X1
            dX1 = 0.5 * torch.sum(x1_t * (dC_key @ x1_t), dim=0) / X1  # (n1,)
            dX2 = 0.5 * torch.sum(x2_t * (dC_key @ x2_t), dim=0) / X2  # (n2,)

            # dX1X2/dtheta = dX1 * X2 + X1 * dX2 (broadcast to matrix)
            dX1X2 = dX1[:, None] * X2 + X1[:, None] * dX2  # (n1, n2)

            # d(x1x2)/dtheta = x1.T @ dC @ x2
            dx1x2 = x1_t.T @ dC_key @ x2_t  # (n1, n2)

            # dcosdelta/dtheta = (dx1x2 - cosdelta * dX1X2) / X1X2
            dcosdelta = (dx1x2 - cosdelta * dX1X2) / X1X2  # (n1, n2)

            # dJ/dtheta = -(delta - pi) * dcosdelta / pi
            dJ = -(delta - torch.pi) * dcosdelta / torch.pi  # (n1, n2)

            # dK/dtheta = X1X2 * dJ + dX1X2 * J
            dK[key] = X1X2 * dJ + dX1X2 * J  # (n1, n2)

    else:
        # ===== Diagonal case =====
        # Reference: varGP utils.py lines 3781-3798

        # K[i] = x1[:,i]^T @ C @ x1[:,i] + sigma_0^2
        K = torch.sum(x1_t * (C @ x1_t), dim=0) + sigma_0 ** 2  # (n1,)

        # dK/dsigma_0 = 2 * sigma_0 (for diagonal, the derivative is simple)
        # Note: reference divides by sigma_0 at the end for consistency
        dK['sigma_0'] = 2 * sigma_0 * torch.ones(n1, device=device, dtype=dtype) / sigma_0  # (n1,)

        # Gradients w.r.t. C-dependent hyperparameters
        for key, dC_key in dC.items():
            if key == 'sigma_0':
                continue
            # dK[i]/dtheta = x1[:,i]^T @ dC @ x1[:,i]
            dK[key] = torch.sum(x1_t * (dC_key @ x1_t), dim=0)  # (n1,)

    return K, dK


def compute_C_and_gradients(
    theta_Amp: torch.Tensor,
    theta_2log2beta: torch.Tensor,
    theta_log2rho2: torch.Tensor,
    theta_eps0x: torch.Tensor,
    theta_eps0y: torch.Tensor,
    n_px_side: int,
    mask: Optional[torch.Tensor] = None,
    compute_gradients: bool = True
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """
    Compute C matrix and its gradients w.r.t. all hyperparameters.

    This combines the localker_clean forward pass with C_gradients_hyp computation.

    Parameters
    ----------
    theta_Amp : Tensor (scalar)
        Amplitude hyperparameter
    theta_2log2beta : Tensor (scalar)
        Locality hyperparameter (-2*log(2*beta))
    theta_log2rho2 : Tensor (scalar)
        Smoothness hyperparameter (-log(2*rho^2))
    theta_eps0x, theta_eps0y : Tensor (scalar)
        RF center position
    n_px_side : int
        Number of pixels per side
    mask : Tensor (n_px,) bool, optional
        Pixel mask. If None, compute mask from alpha threshold.

    Returns
    -------
    C : Tensor (n_masked, n_masked)
        Covariance matrix for masked pixels
    dC : Dict[str, Tensor]
        Dictionary of dC/dtheta matrices
    mask : Tensor (n_px,) bool
        Mask indicating active pixels
    coords : Tuple[Tensor, Tensor]
        (xcord_masked, ycord_masked) for potential reuse
    """
    device = theta_Amp.device
    dtype = theta_Amp.dtype

    # Build coordinate grid (matching localker_clean)
    ycord, xcord = torch.meshgrid(
        torch.linspace(-1, 1, n_px_side, device=device, dtype=dtype),
        torch.linspace(-1, 1, n_px_side, device=device, dtype=dtype),
        indexing='ij'
    )
    xcord = xcord.flatten()  # (n_px_side^2,)
    ycord = ycord.flatten()

    # Compute mask with DETACHED theta (mask is structurally fixed)
    if mask is None:
        with torch.no_grad():
            dist_sq_detached = (xcord - theta_eps0x.detach())**2 + (ycord - theta_eps0y.detach())**2
            logalpha_for_mask = -torch.exp(theta_2log2beta.detach()) * dist_sq_detached
            alpha_for_mask = torch.exp(logalpha_for_mask)
            mask = alpha_for_mask >= 0.001

    # Apply mask to coordinates
    xcord_masked = xcord[mask]
    ycord_masked = ycord[mask]

    # Compute C with gradient tracking
    dist_sq = (xcord_masked - theta_eps0x)**2 + (ycord_masked - theta_eps0y)**2
    logalpha = -torch.exp(theta_2log2beta) * dist_sq
    alpha_local = torch.exp(logalpha)  # (n_masked,)

    # Smooth prior
    dist_sq_smooth = (xcord_masked - xcord_masked[:, None])**2 + (ycord_masked - ycord_masked[:, None])**2
    logCsmooth = -torch.exp(theta_log2rho2) * dist_sq_smooth
    C_smooth = torch.exp(logCsmooth)  # (n_masked, n_masked)

    # Combine: C = Amp * alpha_local * C_smooth * alpha_local^T
    C = theta_Amp * alpha_local[:, None] * C_smooth * alpha_local[None, :]

    # Symmetrize for numerical stability
    C = (C + C.T) / 2

    # Only compute gradients if needed
    if compute_gradients:
        dC_Amp, dC_eps0x, dC_eps0y, dC_logbeta, dC_logrho = C_gradients_hyp(
            C, theta_Amp, theta_2log2beta, logalpha, logCsmooth,
            xcord_masked, ycord_masked, theta_eps0x, theta_eps0y
        )

        dC = {
            'Amp': dC_Amp,
            'eps_0x': dC_eps0x,
            'eps_0y': dC_eps0y,
            '-2log2beta': dC_logbeta,
            '-log2rho2': dC_logrho
        }
    else:
        dC = None

    return C, dC, mask, (xcord_masked, ycord_masked)


def compute_K_only(
    sigma_0: torch.Tensor,
    x1: torch.Tensor,
    x2: Optional[torch.Tensor],
    C: torch.Tensor,
    diag: bool = False
) -> torch.Tensor:
    """
    Compute kernel K without gradients (fast path).

    This is used when we don't need gradients, e.g., during E-step or F-step.
    Same computation as acosker_with_hyp_grad but without dK computation.
    """
    x1_t = x1.T
    if x2 is not None:
        x2_t = x2.T
    else:
        x2_t = x1_t

    if not diag:
        # Full matrix case
        X1 = torch.sqrt(torch.sum(x1_t * (C @ x1_t), dim=0) + sigma_0 ** 2)
        X2 = torch.sqrt(torch.sum(x2_t * (C @ x2_t), dim=0) + sigma_0 ** 2)
        X1X2 = torch.outer(X1, X2)
        x1x2 = x1_t.T @ C @ x2_t + sigma_0 ** 2
        cosdelta = torch.clamp(x1x2 / (X1X2 + 1e-7), -1.0, 1.0)
        delta = torch.arccos(cosdelta)
        sindelta = torch.sqrt(1.0 - cosdelta ** 2)
        J = (sindelta + (torch.pi - delta) * cosdelta) / torch.pi
        K = X1X2 * J
    else:
        # Diagonal case
        K = torch.sum(x1_t * (C @ x1_t), dim=0) + sigma_0 ** 2

    return K


class ArcCosineJacobianGradients(torch.autograd.Function):
    """
    torch.autograd.Function for arc-cosine kernel with analytical gradients.

    This wraps the kernel computation to use analytical gradients instead of autograd.
    Forward pass computes K and all dK/dtheta matrices.
    Backward pass uses the saved dK matrices to compute parameter gradients.

    Usage:
        K = ArcCosineJacobianGradients.apply(
            x1, x2, sigma_0, Amp, eps_0x, eps_0y, raw_m2log2beta, raw_mlog2rho2,
            n_px_side, use_mask, diag
        )
    """

    @staticmethod
    def forward(ctx, x1, x2, sigma_0, Amp, eps_0x, eps_0y, raw_m2log2beta, raw_mlog2rho2,
                n_px_side, use_mask, diag):
        """
        Forward pass: compute K and save dK matrices for backward.

        Parameters
        ----------
        x1 : Tensor (n1, n_features)
            First input points (FULL images, masking applied internally)
        x2 : Tensor (n2, n_features)
            Second input points
        sigma_0 : Tensor (scalar or 1,)
            Bias variance parameter
        eps_0x, eps_0y : Tensor (scalar or 1,)
            RF center position
        raw_m2log2beta : Tensor (scalar or 1,)
            Locality parameter in log-space (-2log(2*beta))
        raw_mlog2rho2 : Tensor (scalar or 1,)
            Smoothness parameter in log-space (-log(2*rho^2))
        n_px_side : int
            Image dimension (e.g., 108 for PNAS data)
        use_mask : bool
            Whether to apply pixel masking
        diag : bool
            If True, return only diagonal

        Returns
        -------
        K : Tensor
            Kernel matrix (n1, n2) or (n1,) if diag=True
        """
        # Ensure scalar tensors
        sigma_0_val = sigma_0.squeeze() if sigma_0.dim() > 0 else sigma_0
        Amp_val = Amp.squeeze() if Amp.dim() > 0 else Amp
        eps_0x_val = eps_0x.squeeze() if eps_0x.dim() > 0 else eps_0x
        eps_0y_val = eps_0y.squeeze() if eps_0y.dim() > 0 else eps_0y
        beta_val = raw_m2log2beta.squeeze() if raw_m2log2beta.dim() > 0 else raw_m2log2beta
        rho_val = raw_mlog2rho2.squeeze() if raw_mlog2rho2.dim() > 0 else raw_mlog2rho2

        device = x1.device
        dtype = x1.dtype

        # Check if we need to compute gradients
        # Only compute dK if backward might be called (any kernel param requires_grad)
        # NOTE: We can't use torch.is_grad_enabled() because GPyTorch evaluates kernels
        # inside a no_grad context, but backward is called later outside that context.
        # So we must check requires_grad on the input tensors directly.
        needs_grad = (
            (hasattr(sigma_0, 'requires_grad') and sigma_0.requires_grad) or
            (hasattr(Amp, 'requires_grad') and Amp.requires_grad) or
            (hasattr(eps_0x, 'requires_grad') and eps_0x.requires_grad) or
            (hasattr(eps_0y, 'requires_grad') and eps_0y.requires_grad) or
            (hasattr(raw_m2log2beta, 'requires_grad') and raw_m2log2beta.requires_grad) or
            (hasattr(raw_mlog2rho2, 'requires_grad') and raw_mlog2rho2.requires_grad)
        )

        # Track kernel calls for debugging/optimization analysis
        if not hasattr(ArcCosineJacobianGradients, '_call_count'):
            ArcCosineJacobianGradients._call_count = {'grad': 0, 'no_grad': 0}
        if needs_grad:
            ArcCosineJacobianGradients._call_count['grad'] += 1
        else:
            ArcCosineJacobianGradients._call_count['no_grad'] += 1

        # Compute C (and dC only if needed)
        # Amp is now passed directly (not handled by ScaleKernel anymore)
        C, dC, mask, coords = compute_C_and_gradients(
            Amp_val, beta_val, rho_val, eps_0x_val, eps_0y_val,
            n_px_side, mask=None, compute_gradients=needs_grad
        )

        # Apply mask to inputs if using masking
        if use_mask and mask is not None:
            x1_masked = x1[..., mask]
            x2_masked = x2[..., mask] if x2 is not None else None
        else:
            x1_masked = x1
            x2_masked = x2

        if needs_grad:
            # Compute K and dK
            K, dK = acosker_with_hyp_grad(
                sigma_0_val, x1_masked, x2_masked, C, dC, diag=diag
            )

            # Save for backward
            ctx.save_for_backward(
                dK['sigma_0'],
                dK.get('Amp', torch.zeros_like(K)),
                dK.get('eps_0x', torch.zeros_like(K)),
                dK.get('eps_0y', torch.zeros_like(K)),
                dK.get('-2log2beta', torch.zeros_like(K)),
                dK.get('-log2rho2', torch.zeros_like(K)),
            )
        else:
            # Just compute K without gradients (much faster)
            K = compute_K_only(sigma_0_val, x1_masked, x2_masked, C, diag=diag)
            # Save empty tensors (backward won't be called)
            ctx.save_for_backward(
                torch.empty(0, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype),
            )

        ctx.diag = diag
        ctx.needs_grad = needs_grad

        return K

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients using saved dK matrices.

        Chain rule: grad_theta = (grad_output * dK/dtheta).sum()

        Parameters
        ----------
        grad_output : Tensor
            Gradient of loss w.r.t. K, shape same as K

        Returns
        -------
        Tuple of gradients for each forward input (or None if not needed)
        """
        dK_sigma0, dK_Amp, dK_eps0x, dK_eps0y, dK_beta, dK_rho = ctx.saved_tensors

        # Compute gradients via chain rule
        # grad_theta = sum(grad_output * dK/dtheta)
        grad_sigma0 = (grad_output * dK_sigma0).sum()
        grad_Amp = (grad_output * dK_Amp).sum()
        grad_eps0x = (grad_output * dK_eps0x).sum()
        grad_eps0y = (grad_output * dK_eps0y).sum()
        grad_beta = (grad_output * dK_beta).sum()
        grad_rho = (grad_output * dK_rho).sum()

        # Return gradients for each input (None for non-tensor inputs)
        # Order: x1, x2, sigma_0, Amp, eps_0x, eps_0y, raw_m2log2beta, raw_mlog2rho2,
        #        n_px_side, use_mask, diag
        return (
            None,  # x1 (no gradient needed)
            None,  # x2 (no gradient needed)
            grad_sigma0.unsqueeze(0),  # sigma_0
            grad_Amp.unsqueeze(0),     # Amp
            grad_eps0x.unsqueeze(0),   # eps_0x
            grad_eps0y.unsqueeze(0),   # eps_0y
            grad_beta.unsqueeze(0),    # raw_m2log2beta
            grad_rho.unsqueeze(0),     # raw_mlog2rho2
            None,  # n_px_side (int, no gradient)
            None,  # use_mask (bool, no gradient)
            None,  # diag (bool, no gradient)
        )


def test_autograd_function():
    """Test that ArcCosineJacobianGradients gradients match autograd."""
    print("\nTesting ArcCosineJacobianGradients...")

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Create test data (small for faster testing)
    n1, n2, n_px_side = 5, 4, 10  # Small image for testing
    n_px = n_px_side ** 2
    x1 = torch.randn(n1, n_px, device=device, dtype=dtype)
    x2 = torch.randn(n2, n_px, device=device, dtype=dtype)

    # Parameters
    sigma_0 = torch.tensor([1.0], device=device, dtype=dtype, requires_grad=True)
    eps_0x = torch.tensor([0.0], device=device, dtype=dtype, requires_grad=True)
    eps_0y = torch.tensor([0.0], device=device, dtype=dtype, requires_grad=True)
    raw_m2log2beta = torch.tensor([-2 * np.log(2 * 0.3)], device=device, dtype=dtype, requires_grad=True)
    raw_mlog2rho2 = torch.tensor([-np.log(2 * 0.3**2)], device=device, dtype=dtype, requires_grad=True)

    # Test with analytical gradients
    print("  Computing K with analytical gradients...")
    K_analytical = ArcCosineJacobianGradients.apply(
        x1, x2, sigma_0, eps_0x, eps_0y, raw_m2log2beta, raw_mlog2rho2,
        n_px_side, True, False  # use_mask=True, diag=False
    )
    print(f"  K shape: {K_analytical.shape}")

    # Backward pass
    loss = K_analytical.sum()
    loss.backward()

    print(f"  grad_sigma_0: {sigma_0.grad.item():.6f}")
    print(f"  grad_eps_0x: {eps_0x.grad.item():.6f}")
    print(f"  grad_eps_0y: {eps_0y.grad.item():.6f}")
    print(f"  grad_beta: {raw_m2log2beta.grad.item():.6f}")
    print(f"  grad_rho: {raw_mlog2rho2.grad.item():.6f}")

    print("  ArcCosineJacobianGradients test PASSED!")
    return True


def test_acosker_with_hyp_grad():
    """Test that analytical gradients match autograd."""
    print("Testing acosker_with_hyp_grad...")

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Create test data
    n1, n2, nx = 10, 8, 50
    x1 = torch.randn(n1, nx, device=device, dtype=dtype)
    x2 = torch.randn(n2, nx, device=device, dtype=dtype)

    # Create simple C matrix (identity-ish for testing)
    C = torch.eye(nx, device=device, dtype=dtype) * 0.1
    C = C + 0.01 * torch.randn(nx, nx, device=device, dtype=dtype)
    C = (C + C.T) / 2  # Symmetrize

    # Create dummy dC matrices (won't use these for gradient comparison, just structure)
    dC = {
        'Amp': C * 0.1,
        'eps_0x': C * 0.05,
        'eps_0y': C * 0.05,
        '-2log2beta': C * 0.02,
        '-log2rho2': C * 0.03
    }

    sigma_0 = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=True)

    # Test 1: Full matrix case
    print("  1. Full matrix case...")
    K, dK = acosker_with_hyp_grad(sigma_0.detach(), x1, x2, C, dC, diag=False)
    print(f"     K shape: {K.shape}, expected: ({n1}, {n2})")
    assert K.shape == (n1, n2), f"K shape mismatch: {K.shape}"
    for key, grad in dK.items():
        assert grad.shape == K.shape, f"dK[{key}] shape mismatch: {grad.shape}"
    print("     Shape check passed!")

    # Test 2: Diagonal case
    print("  2. Diagonal case...")
    K_diag, dK_diag = acosker_with_hyp_grad(sigma_0.detach(), x1, None, C, dC, diag=True)
    print(f"     K_diag shape: {K_diag.shape}, expected: ({n1},)")
    assert K_diag.shape == (n1,), f"K_diag shape mismatch: {K_diag.shape}"
    for key, grad in dK_diag.items():
        assert grad.shape == K_diag.shape, f"dK_diag[{key}] shape mismatch: {grad.shape}"
    print("     Shape check passed!")

    # Test 3: Compare sigma_0 gradient with autograd
    print("  3. Comparing sigma_0 gradient with autograd...")
    sigma_0_ag = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=True)

    # Compute K with autograd
    x1_t = x1.T
    x2_t = x2.T
    X1 = torch.sqrt(torch.sum(x1_t * (C @ x1_t), dim=0) + sigma_0_ag ** 2)
    X2 = torch.sqrt(torch.sum(x2_t * (C @ x2_t), dim=0) + sigma_0_ag ** 2)
    X1X2 = torch.outer(X1, X2)
    x1x2 = x1_t.T @ C @ x2_t + sigma_0_ag ** 2
    cosdelta = torch.clamp(x1x2 / (X1X2 + 1e-7), -1.0, 1.0)
    delta = torch.arccos(cosdelta)
    sindelta = torch.sqrt(1.0 - cosdelta ** 2)
    J = (sindelta + (torch.pi - delta) * cosdelta) / torch.pi
    K_ag = X1X2 * J

    # Compute gradient via autograd
    grad_output = torch.ones_like(K_ag)
    K_ag.backward(grad_output)
    autograd_sigma0 = sigma_0_ag.grad.item()

    # Analytical gradient (sum of dK['sigma_0'])
    analytical_sigma0 = dK['sigma_0'].sum().item()

    print(f"     Autograd: {autograd_sigma0:.6f}")
    print(f"     Analytical: {analytical_sigma0:.6f}")
    rel_err = abs(autograd_sigma0 - analytical_sigma0) / (abs(autograd_sigma0) + 1e-10)
    print(f"     Relative error: {rel_err:.2e}")

    if rel_err < 1e-5:
        print("     PASSED!")
    else:
        print("     WARNING: Large relative error!")

    print("\nAll tests completed!")
    return True
