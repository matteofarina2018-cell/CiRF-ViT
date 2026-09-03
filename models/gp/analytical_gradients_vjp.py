"""
VJP-Based Analytical Gradients for Arc-Cosine Kernel (EXPERIMENTAL)

This implements efficient analytical gradients using Vector-Jacobian Products (VJPs)
instead of materializing full Jacobian matrices.

Key difference from analytical_gradients.py:
- OLD: Materialize 5 dC matrices, 5 dK matrices, 15 large matrix multiplies
- NEW: Compute dL/dC ONCE in backward, then chain to hyperparameters via element-wise ops

Expected speedup: ~10-15x for gradient computation. The derivation is that of
Goldin, Virgili and Chalk (PNAS 2023), rearranged so that dL/dC is formed once.
"""

import torch
from typing import Tuple, Optional


class ArcCosineVJPGradients(torch.autograd.Function):
    """
    VJP-based arc-cosine kernel with analytical gradients.

    Forward: Compute K and save intermediates for backward.
    Backward: Use VJP chain to compute gradients efficiently.

    Key insight: Instead of computing dK/dθ for each θ (expensive),
    we compute dL/dC once and chain through to each θ (cheap).
    """

    @staticmethod
    def forward(ctx, x1, x2, sigma_0, Amp, eps_0x, eps_0y, raw_m2log2beta, raw_mlog2rho2,
                n_px_side, use_mask, diag):
        """
        Forward pass: compute K and save intermediates.

        We save everything needed for backward, but do NOT compute any gradient matrices.
        """
        # Squeeze scalar parameters
        sigma_0_val = sigma_0.squeeze() if sigma_0.dim() > 0 else sigma_0
        Amp_val = Amp.squeeze() if Amp.dim() > 0 else Amp
        eps_0x_val = eps_0x.squeeze() if eps_0x.dim() > 0 else eps_0x
        eps_0y_val = eps_0y.squeeze() if eps_0y.dim() > 0 else eps_0y
        beta_raw = raw_m2log2beta.squeeze() if raw_m2log2beta.dim() > 0 else raw_m2log2beta
        rho_raw = raw_mlog2rho2.squeeze() if raw_mlog2rho2.dim() > 0 else raw_mlog2rho2

        device = x1.device
        dtype = x1.dtype

        # Check if we need gradients
        needs_grad = (
            (hasattr(sigma_0, 'requires_grad') and sigma_0.requires_grad) or
            (hasattr(Amp, 'requires_grad') and Amp.requires_grad) or
            (hasattr(eps_0x, 'requires_grad') and eps_0x.requires_grad) or
            (hasattr(eps_0y, 'requires_grad') and eps_0y.requires_grad) or
            (hasattr(raw_m2log2beta, 'requires_grad') and raw_m2log2beta.requires_grad) or
            (hasattr(raw_mlog2rho2, 'requires_grad') and raw_mlog2rho2.requires_grad)
        )

        # Track kernel calls for debugging
        if not hasattr(ArcCosineVJPGradients, '_call_count'):
            ArcCosineVJPGradients._call_count = {'grad': 0, 'no_grad': 0}
        if needs_grad:
            ArcCosineVJPGradients._call_count['grad'] += 1
        else:
            ArcCosineVJPGradients._call_count['no_grad'] += 1

        # ===== Stage 1: Build coordinate grid and mask =====
        ycord, xcord = torch.meshgrid(
            torch.linspace(-1, 1, n_px_side, device=device, dtype=dtype),
            torch.linspace(-1, 1, n_px_side, device=device, dtype=dtype),
            indexing='ij'
        )
        xcord = xcord.flatten()
        ycord = ycord.flatten()

        # Compute mask with detached parameters (structural stability)
        if use_mask:
            with torch.no_grad():
                dist_sq_det = (xcord - eps_0x_val.detach())**2 + (ycord - eps_0y_val.detach())**2
                beta_det = torch.exp(beta_raw.detach())
                logalpha_det = -beta_det * dist_sq_det
                alpha_det = torch.exp(logalpha_det)
                mask = alpha_det >= 0.001
            xcord = xcord[mask]
            ycord = ycord[mask]
        else:
            mask = None

        nx = xcord.shape[0]  # Number of (masked) pixels

        # ===== Stage 2: Compute C matrix components =====
        # Transform parameters: beta = exp(raw), rho2 = exp(raw)
        beta = torch.exp(beta_raw)
        rho2 = torch.exp(rho_raw)

        # Distance from RF center
        dist_center = (xcord - eps_0x_val)**2 + (ycord - eps_0y_val)**2  # (nx,)

        # Alpha (locality weights)
        logalpha = -beta * dist_center
        alpha = torch.exp(logalpha)  # (nx,)

        # Pairwise distances (for C_smooth)
        dist_pairwise = (xcord[:, None] - xcord[None, :])**2 + (ycord[:, None] - ycord[None, :])**2  # (nx, nx)

        # C_smooth matrix
        logS = -rho2 * dist_pairwise
        S = torch.exp(logS)  # (nx, nx)

        # Full C matrix with Amp scaling (matches legacy varGP)
        # C = Amp * alpha * C_smooth * alpha^T
        C = Amp_val * alpha[:, None] * S * alpha[None, :]  # (nx, nx)
        C = (C + C.T) / 2  # Symmetrize for numerical stability

        # ===== Stage 3: Apply mask to inputs =====
        if mask is not None:
            x1_masked = x1[..., mask]
            x2_masked = x2[..., mask] if x2 is not None else None
        else:
            x1_masked = x1
            x2_masked = x2

        # Transpose to (nx, n) convention
        x1_t = x1_masked.T  # (nx, n1)
        x2_t = x2_masked.T if x2_masked is not None else x1_t  # (nx, n2)

        n1 = x1_t.shape[1]
        n2 = x2_t.shape[1]
        sigma_0_sq = sigma_0_val ** 2

        # ===== Stage 4: Compute K =====
        if not diag:
            # Cx1, Cx2 needed for backward
            Cx1 = C @ x1_t  # (nx, n1)
            Cx2 = C @ x2_t  # (nx, n2)

            # V1[i] = x1[:,i]^T @ C @ x1[:,i] + sigma_0^2
            V1 = torch.sum(x1_t * Cx1, dim=0) + sigma_0_sq  # (n1,)
            V2 = torch.sum(x2_t * Cx2, dim=0) + sigma_0_sq  # (n2,)

            X1 = torch.sqrt(V1)  # (n1,)
            X2 = torch.sqrt(V2)  # (n2,)
            X1X2 = torch.outer(X1, X2)  # (n1, n2)

            # Cross term: x1Cx2[i,j] = x1[:,i]^T @ C @ x2[:,j] + sigma_0^2
            x1Cx2 = x1_t.T @ Cx2 + sigma_0_sq  # (n1, n2)

            # Normalized inner product (clamp for stability)
            eps = 1e-7
            cosdelta = torch.clamp(x1Cx2 / (X1X2 + eps), -1.0 + eps, 1.0 - eps)

            # Angles
            delta = torch.arccos(cosdelta)
            sindelta = torch.sqrt(torch.clamp(1.0 - cosdelta**2, min=eps))

            # J function
            J = (sindelta + (torch.pi - delta) * cosdelta) / torch.pi

            # Kernel
            K = X1X2 * J

            if needs_grad:
                # Save all intermediates for backward
                ctx.save_for_backward(
                    # K computation intermediates
                    x1_t, x2_t, Cx1, Cx2,
                    X1, X2, X1X2, V1, V2,
                    cosdelta, delta, J, x1Cx2,
                    # C computation intermediates
                    alpha, S, C,
                    dist_center, dist_pairwise,
                    xcord, ycord,
                    # Parameters
                    sigma_0_val, Amp_val, eps_0x_val, eps_0y_val, beta_raw, rho_raw
                )
                ctx.n1 = n1
                ctx.n2 = n2
                ctx.nx = nx
                ctx.diag = False
                # Save original input shapes for gradient reshaping
                ctx.sigma0_shape = sigma_0.shape
                ctx.Amp_shape = Amp.shape
                ctx.eps0x_shape = eps_0x.shape
                ctx.eps0y_shape = eps_0y.shape
                ctx.beta_shape = raw_m2log2beta.shape
                ctx.rho_shape = raw_mlog2rho2.shape
            else:
                ctx.save_for_backward(*([torch.empty(0, device=device, dtype=dtype)] * 25))
                ctx.diag = False

            ctx.needs_grad = needs_grad
            return K

        else:
            # Diagonal case: K[i] = V1[i] (cosdelta=1, J=1, so K = X1*X1 = V1)
            Cx1 = C @ x1_t
            V1 = torch.sum(x1_t * Cx1, dim=0) + sigma_0_sq
            K = V1

            if needs_grad:
                ctx.save_for_backward(
                    x1_t, x1_t, Cx1, Cx1,
                    torch.sqrt(V1), torch.sqrt(V1), V1, V1, V1,
                    torch.ones_like(V1), torch.zeros_like(V1), torch.ones_like(V1), V1,
                    alpha, S, C,
                    dist_center, dist_pairwise,
                    xcord, ycord,
                    sigma_0_val, Amp_val, eps_0x_val, eps_0y_val, beta_raw, rho_raw
                )
                ctx.n1 = n1
                ctx.n2 = n1
                ctx.nx = nx
                ctx.diag = True
                # Save original input shapes for gradient reshaping
                ctx.sigma0_shape = sigma_0.shape
                ctx.Amp_shape = Amp.shape
                ctx.eps0x_shape = eps_0x.shape
                ctx.eps0y_shape = eps_0y.shape
                ctx.beta_shape = raw_m2log2beta.shape
                ctx.rho_shape = raw_mlog2rho2.shape
            else:
                ctx.save_for_backward(*([torch.empty(0, device=device, dtype=dtype)] * 25))
                ctx.diag = True

            ctx.needs_grad = needs_grad
            return K

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: VJP chain to compute gradients efficiently.

        Given dL/dK (grad_output), compute dL/d(each parameter).

        Key insight: We compute dL/dC ONCE, then extract all hyperparameter
        gradients via element-wise operations with dL/dC.
        """
        if not ctx.needs_grad:
            return (None,) * 11

        # Retrieve saved tensors
        (x1_t, x2_t, Cx1, Cx2,
         X1, X2, X1X2, V1, V2,
         cosdelta, delta, J, x1Cx2,
         alpha, S, C,
         dist_center, dist_pairwise,
         xcord, ycord,
         sigma_0_val, Amp_val, eps_0x_val, eps_0y_val, beta_raw, rho_raw) = ctx.saved_tensors

        n1, n2, nx = ctx.n1, ctx.n2, ctx.nx
        G = grad_output  # dL/dK, shape (n1, n2) or (n1,) for diag
        eps = 1e-7

        # Transform parameters
        beta = torch.exp(beta_raw)
        rho2 = torch.exp(rho_raw)

        if ctx.diag:
            # ===== Diagonal case: K = V1 =====
            # dL/dV1 = G directly
            dL_dV1 = G  # (n1,)
            dL_dV2 = torch.zeros_like(G)
            dL_dx1Cx2 = torch.zeros(n1, n1, device=G.device, dtype=G.dtype)

            # dL/dsigma_0: V1 = ... + sigma_0^2
            grad_sigma0 = 2 * sigma_0_val * dL_dV1.sum()

        else:
            # ===== Full matrix case: VJP chain through K → C =====

            # Step 1: dL/dK → dL/dJ, dL/dX1X2
            dL_dJ = G * X1X2           # (n1, n2)
            dL_dX1X2_K = G * J          # (n1, n2)

            # Step 2: dL/dJ → dL/dcosdelta
            # J = (sin + (π-δ)*cos)/π, and dJ/dcos = (π-δ)/π
            dL_dcosdelta = dL_dJ * (torch.pi - delta) / torch.pi  # (n1, n2)

            # Step 3: dL/dcosdelta → dL/dx1Cx2, dL/dX1X2
            # cosdelta = x1Cx2 / X1X2
            dL_dx1Cx2 = dL_dcosdelta / (X1X2 + eps)  # (n1, n2)
            dL_dX1X2_cos = -dL_dcosdelta * cosdelta / (X1X2 + eps)  # (n1, n2)

            dL_dX1X2 = dL_dX1X2_K + dL_dX1X2_cos  # (n1, n2)

            # Step 4: dL/dX1X2 → dL/dX1, dL/dX2
            # X1X2[i,j] = X1[i] * X2[j]
            dL_dX1 = dL_dX1X2 @ X2   # (n1,)
            dL_dX2 = dL_dX1X2.T @ X1  # (n2,)

            # Step 5: dL/dX1 → dL/dV1, dL/dV2
            # X1 = sqrt(V1), so dX1/dV1 = 1/(2*X1)
            dL_dV1 = dL_dX1 / (2 * X1 + eps)  # (n1,)
            dL_dV2 = dL_dX2 / (2 * X2 + eps)  # (n2,)

            # Step 6a: dL/dsigma_0 (sigma_0 appears directly in V1, V2, x1Cx2)
            # V1 = ... + σ₀², V2 = ... + σ₀², x1Cx2 = ... + σ₀²
            grad_sigma0 = 2 * sigma_0_val * (dL_dV1.sum() + dL_dV2.sum() + dL_dx1Cx2.sum())

        # ===== Step 6b: Compute dL/dC (the key efficiency gain) =====
        # V1[i] = x1[:,i]ᵀ C x1[:,i], so dL/dC from V1 = Σᵢ dL/dV1[i] * x1[:,i] ⊗ x1[:,i]
        # In matrix form: x1_t @ diag(dL/dV1) @ x1_t.T = (x1_t * dL_dV1) @ x1_t.T

        dL_dC = (x1_t * dL_dV1) @ x1_t.T  # (nx, nx)
        dL_dC = dL_dC + (x2_t * dL_dV2) @ x2_t.T  # Add V2 contribution

        if not ctx.diag:
            # Add x1Cx2 contribution: x1_t @ dL/dx1Cx2 @ x2_t.T
            dL_dC = dL_dC + x1_t @ dL_dx1Cx2 @ x2_t.T

        # Symmetrize (since C is symmetric, dL/dC should be too)
        dL_dC = (dL_dC + dL_dC.T) / 2

        # ===== Step 7: Chain dL/dC to hyperparameters =====
        # C = Amp * alpha[:,None] * S * alpha[None,:]

        # dL/dAmp: C = Amp * C_base, so dC/dAmp = C_base = C/Amp
        # grad_Amp = sum(dL/dC * dC/dAmp) = sum(dL/dC * C) / Amp
        # This matches legacy varGP: dC_Amp = C / theta['Amp']
        grad_Amp = (dL_dC * C).sum() / Amp_val

        # dL/dalpha: A = alpha ⊗ alpha, so dL/dA = dL/dC * S * Amp
        # For A = alpha ⊗ alpha: dL/dalpha = 2 * (dL/dA @ alpha)
        dL_dA = dL_dC * S * Amp_val  # (nx, nx)
        dL_dalpha = 2 * (dL_dA @ alpha)  # (nx,)

        # dL/dbeta_raw (through alpha)
        # alpha = exp(-beta * dist_center), where beta = exp(beta_raw)
        # dalpha/d(beta_raw) = alpha * (-dist_center) * beta
        grad_beta = (dL_dalpha * alpha * (-dist_center) * beta).sum()

        # dL/deps_0x, dL/deps_0y (through alpha via dist_center)
        # dist_center = (x - eps_0x)² + (y - eps_0y)²
        # d(dist_center)/d(eps_0x) = -2*(x - eps_0x)
        # dalpha/d(eps_0x) = alpha * (-beta) * d(dist_center)/d(eps_0x) = alpha * 2*beta*(x - eps_0x)
        grad_eps0x = (dL_dalpha * alpha * 2 * beta * (xcord - eps_0x_val)).sum()
        grad_eps0y = (dL_dalpha * alpha * 2 * beta * (ycord - eps_0y_val)).sum()

        # dL/drho_raw (through S)
        # S = exp(-rho2 * dist_pairwise), where rho2 = exp(rho_raw)
        # dL/dS = dL/dC * Amp * alpha[:,None] * alpha[None,:] (since C = Amp*α*S*α)
        A = alpha[:, None] * alpha[None, :]
        dL_dS = dL_dC * Amp_val * A  # (nx, nx)

        # dS/d(rho_raw) = S * (-dist_pairwise) * rho2
        grad_rho = (dL_dS * S * (-dist_pairwise) * rho2).sum()

        # Return gradients for all forward inputs
        # Note: gradients must match the shape of inputs (may be [1] or scalar)
        return (
            None,  # x1
            None,  # x2
            grad_sigma0.reshape(ctx.sigma0_shape),
            grad_Amp.reshape(ctx.Amp_shape),
            grad_eps0x.reshape(ctx.eps0x_shape),
            grad_eps0y.reshape(ctx.eps0y_shape),
            grad_beta.reshape(ctx.beta_shape),
            grad_rho.reshape(ctx.rho_shape),
            None,  # n_px_side
            None,  # use_mask
            None,  # diag
        )


def test_vjp_correctness():
    """Test that VJP gradients match autograd exactly."""
    import sys
    sys.path.insert(0, '.')
    from .kernels import ArcCosineKernel

    print("=" * 60)
    print("Testing VJP Gradient Correctness")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Small test case
    n1, n2, n_px_side = 10, 8, 16
    n_px = n_px_side ** 2
    x1 = torch.randn(n1, n_px, device=device, dtype=dtype)
    x2 = torch.randn(n2, n_px, device=device, dtype=dtype)

    config = {'sigma_0': 1.0, 'Amp': 1e-4, 'beta': 0.1, 'rho': 0.1, 'eps_0x': 0.0, 'eps_0y': 0.0}

    # Autograd reference
    kernel_auto = ArcCosineKernel(
        sigma_0=config['sigma_0'],
        Amp=config['Amp'],
        n_px_side=n_px_side,
        eps_0x=config['eps_0x'],
        eps_0y=config['eps_0y'],
        beta=config['beta'],
        rho=config['rho'],
        gradient_mode='autograd'
    ).to(device).double()

    K_auto = kernel_auto(x1, x2).to_dense()
    loss_auto = K_auto.sum()
    loss_auto.backward()

    # Get raw gradients from autograd
    # Note: sigma_0 uses exp constraint, Amp uses softplus constraint
    # when comparing with VJP which operates on constrained values directly
    raw_sigma0 = kernel_auto.raw_sigma_0.clone().detach()
    raw_Amp = kernel_auto.raw_Amp.clone().detach()

    auto_grads = {
        'raw_sigma_0': kernel_auto.raw_sigma_0.grad.item(),
        'raw_Amp': kernel_auto.raw_Amp.grad.item(),
        'eps_0x': kernel_auto.eps_0x.grad.item(),
        'eps_0y': kernel_auto.eps_0y.grad.item(),
        'beta': kernel_auto.raw_m2log2beta.grad.item(),
        'rho': kernel_auto.raw_mlog2rho2.grad.item(),
    }

    # VJP implementation - we pass constrained sigma_0 and Amp directly
    # The VJP computes dL/d(sigma_0), not dL/d(raw_sigma_0)
    # To compare: dL/d(raw_sigma_0) = dL/d(sigma_0) * d(sigma_0)/d(raw_sigma_0)
    # For exp: d(exp(x))/dx = exp(x) = sigma_0
    sigma_0 = kernel_auto.sigma_0.clone().detach().requires_grad_(True)
    Amp = kernel_auto.Amp.clone().detach().requires_grad_(True)
    eps_0x = kernel_auto.eps_0x.clone().detach().requires_grad_(True)
    eps_0y = kernel_auto.eps_0y.clone().detach().requires_grad_(True)
    raw_beta = kernel_auto.raw_m2log2beta.clone().detach().requires_grad_(True)
    raw_rho = kernel_auto.raw_mlog2rho2.clone().detach().requires_grad_(True)

    K_vjp = ArcCosineVJPGradients.apply(
        x1, x2,
        sigma_0, Amp, eps_0x, eps_0y, raw_beta, raw_rho,
        n_px_side, False, False  # use_mask=False, diag=False
    )
    loss_vjp = K_vjp.sum()
    loss_vjp.backward()

    # Convert VJP gradients to raw gradients for comparison
    # sigma_0: identity transform -> d(raw)/d(raw) = 1.0 (no conversion needed)
    # Amp: softplus transform -> d(softplus(raw))/d(raw) = sigmoid(raw)
    sigmoid_raw_Amp = torch.sigmoid(raw_Amp).item()
    vjp_grad_raw_sigma0 = sigma_0.grad.item() if sigma_0.grad is not None else 0
    vjp_grad_raw_Amp = Amp.grad.item() * sigmoid_raw_Amp if Amp.grad is not None else 0

    vjp_grads = {
        'raw_sigma_0': vjp_grad_raw_sigma0,
        'raw_Amp': vjp_grad_raw_Amp,
        'eps_0x': eps_0x.grad.item() if eps_0x.grad is not None else 0,
        'eps_0y': eps_0y.grad.item() if eps_0y.grad is not None else 0,
        'beta': raw_beta.grad.item() if raw_beta.grad is not None else 0,
        'rho': raw_rho.grad.item() if raw_rho.grad is not None else 0,
    }

    print(f"\nK matrix max diff: {(K_auto - K_vjp).abs().max().item():.2e}")
    print(f"Constraint derivatives: sigma_0=identity (factor 1.0), sigmoid(raw_Amp)={sigmoid_raw_Amp:.4f} (softplus)")
    print("\nGradient comparison (autograd vs VJP):")

    all_pass = True
    for name in ['raw_sigma_0', 'raw_Amp', 'eps_0x', 'eps_0y', 'beta', 'rho']:
        auto = auto_grads[name]
        vjp = vjp_grads[name]
        rel_err = abs(auto - vjp) / (abs(auto) + 1e-10)
        status = "PASS" if rel_err < 1e-4 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name:12s}: autograd={auto:12.4f}, vjp={vjp:12.4f}, rel_err={rel_err:.2e} [{status}]")

    print("\n" + "=" * 60)
    print("RESULT:", "ALL TESTS PASSED!" if all_pass else "SOME TESTS FAILED!")
    print("=" * 60)

    return all_pass


def benchmark_implementations():
    """Benchmark VJP vs current analytical vs autograd."""
    import time
    import sys
    sys.path.insert(0, '.')
    from .kernels import ArcCosineKernel
    from .analytical_gradients import ArcCosineJacobianGradients

    print("\n" + "=" * 60)
    print("Benchmarking: VJP vs Current Analytical vs Autograd")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Realistic PNAS-like sizes
    n1, n2, n_px_side = 500, 50, 108
    n_px = n_px_side ** 2
    x1 = torch.randn(n1, n_px, device=device, dtype=dtype)
    x2 = torch.randn(n2, n_px, device=device, dtype=dtype)

    n_warmup = 2
    n_trials = 5

    def make_params(requires_grad=True):
        return (
            torch.tensor(1.0, device=device, dtype=dtype, requires_grad=requires_grad),  # sigma_0
            torch.tensor(1e-4, device=device, dtype=dtype, requires_grad=requires_grad),  # Amp
            torch.tensor(0.0, device=device, dtype=dtype, requires_grad=requires_grad),  # eps_0x
            torch.tensor(0.0, device=device, dtype=dtype, requires_grad=requires_grad),  # eps_0y
            torch.tensor(3.22, device=device, dtype=dtype, requires_grad=requires_grad),  # beta_raw
            torch.tensor(3.91, device=device, dtype=dtype, requires_grad=requires_grad),  # rho_raw
        )

    # ===== Benchmark Autograd =====
    print("\n1. Autograd (no analytical gradients):")
    kernel_auto = ArcCosineKernel(
        sigma_0=1.0, Amp=1e-4, n_px_side=n_px_side, beta=0.1, rho=0.1,
        use_mask=True, gradient_mode='autograd'
    ).to(device).double()

    for _ in range(n_warmup):
        kernel_auto.zero_grad()
        K = kernel_auto(x1, x2).to_dense()
        K.sum().backward()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_trials):
        kernel_auto.zero_grad()
        K = kernel_auto(x1, x2).to_dense()
        K.sum().backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_autograd = (time.time() - t0) / n_trials
    print(f"   Time per call: {t_autograd*1000:.1f} ms")

    # ===== Benchmark Current Analytical =====
    print("\n2. Current Analytical (Jacobian materialization):")
    params = make_params()

    for _ in range(n_warmup):
        for p in params:
            p.grad = None
        K = ArcCosineJacobianGradients.apply(x1, x2, *params, n_px_side, True, False)
        K.sum().backward()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_trials):
        for p in params:
            p.grad = None
        K = ArcCosineJacobianGradients.apply(x1, x2, *params, n_px_side, True, False)
        K.sum().backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_current = (time.time() - t0) / n_trials
    print(f"   Time per call: {t_current*1000:.1f} ms")

    # ===== Benchmark VJP =====
    print("\n3. VJP (new implementation):")
    params = make_params()

    for _ in range(n_warmup):
        for p in params:
            p.grad = None
        K = ArcCosineVJPGradients.apply(x1, x2, *params, n_px_side, True, False)
        K.sum().backward()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_trials):
        for p in params:
            p.grad = None
        K = ArcCosineVJPGradients.apply(x1, x2, *params, n_px_side, True, False)
        K.sum().backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_vjp = (time.time() - t0) / n_trials
    print(f"   Time per call: {t_vjp*1000:.1f} ms")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Autograd:           {t_autograd*1000:6.1f} ms  (baseline)")
    print(f"  Current Analytical: {t_current*1000:6.1f} ms  ({t_current/t_autograd:.1f}x autograd)")
    print(f"  VJP:                {t_vjp*1000:6.1f} ms  ({t_vjp/t_autograd:.1f}x autograd)")
    print(f"\n  VJP speedup vs Current: {t_current/t_vjp:.1f}x")

    return t_autograd, t_current, t_vjp
