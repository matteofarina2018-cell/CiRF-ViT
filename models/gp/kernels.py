"""
Arc-Cosine Kernel for GPyTorch with Receptive Field Structure

This module implements the arc-cosine kernel as a GPyTorch kernel class.
The arc-cosine kernel is derived from infinite-width 2-layer ReLU neural networks.
The C matrix is computed from receptive field parameters (β, ρ, ξ₀).

Mathematical definition:
    K(x, x') = (1/π) · M · J(θ)

    where:
        v_x     = xᵀCx + σ₀²
        v_x'    = x'ᵀCx' + σ₀²
        M       = √(v_x · v_x')
        cos(θ)  = (xᵀCx' + σ₀²) / M
        J(θ)    = sin(θ) + (π - θ)cos(θ)

The C matrix is computed from RF parameters:
    C = Amp · α · C_smooth · αᵀ
    where α is the locality mask and C_smooth captures spatial correlations.

Gradient Modes:
    - 'autograd': PyTorch autograd (default) - automatic differentiation
    - 'vjp': VJP analytical gradients - same speed as autograd, explicit formulas
    - 'jacobian': Old Jacobian materialization - slow but matches original varGP exactly

Reference: kernels/kernels.py:acosker_clean(), localker_clean()
"""

import warnings

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive

# Valid gradient modes
GRADIENT_MODES = ('autograd', 'vjp', 'jacobian')

# Lazy imports for gradient implementations (avoid circular deps)
_ArcCosineJacobianGradients = None  # Jacobian implementation
_ArcCosineVJPGradients = None       # VJP implementation


def _get_jacobian_implementation():
    """Lazy import for Jacobian-based analytical gradients (slow, reference)."""
    global _ArcCosineJacobianGradients
    if _ArcCosineJacobianGradients is None:
        from .analytical_gradients import ArcCosineJacobianGradients
        _ArcCosineJacobianGradients = ArcCosineJacobianGradients
    return _ArcCosineJacobianGradients


def _get_vjp_implementation():
    """Lazy import for VJP-based analytical gradients (fast, recommended)."""
    global _ArcCosineVJPGradients
    if _ArcCosineVJPGradients is None:
        from .analytical_gradients_vjp import ArcCosineVJPGradients
        _ArcCosineVJPGradients = ArcCosineVJPGradients
    return _ArcCosineVJPGradients


# =========================================================================
# Kernel factory
# =========================================================================

KERNEL_TYPES = ('arc_cosine', 'arc_sine', 'rbf')


def create_kernel(config, n_px_side, eps_0x, eps_0y):
    """Create the appropriate kernel based on config['kernel_type'].

    All kernels share the same RF structure (C matrix, masking, center bounds).
    LocalRBFKernel additionally requires a lengthscale parameter.

    Args:
        config: Flat config dict with kernel_type, sigma_0, Amp, beta, rho,
                use_mask, gradient_mode, and lengthscale (for rbf).
        n_px_side: Image side length in pixels.
        eps_0x, eps_0y: RF center coordinates (normalized).
    """
    kernel_type = config['kernel_type']
    common = dict(
        n_px_side=n_px_side,
        sigma_0=config['sigma_0'],
        Amp=config['Amp'],
        eps_0x=eps_0x,
        eps_0y=eps_0y,
        beta=config['beta'],
        rho=config['rho'],
        use_mask=config['use_mask'],
        gradient_mode=config['gradient_mode'],
    )
    if kernel_type == 'arc_cosine':
        return ArcCosineKernel(**common)
    elif kernel_type == 'arc_sine':
        return ArcSineKernel(**common)
    elif kernel_type == 'rbf':
        return LocalRBFKernel(**common, lengthscale=config['lengthscale'])
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. "
                         f"Supported: {KERNEL_TYPES}")


# =========================================================================
# Kernel classes
# =========================================================================

class ArcCosineKernel(Kernel):
    """Arc-cosine kernel for GPyTorch with receptive field structure.

    This kernel is non-stationary (depends on actual input values, not just distances).
    It's derived from infinite-width 2-layer neural networks with ReLU activations.
    The kernel computes covariance matrices from receptive field parameters.

    Parameters
    ----------
    n_px_side : int
        Image dimension (e.g., 108 for PNAS data). Required - specifies the
        spatial structure for computing the C matrix from RF parameters.
    sigma_0 : float, optional
        Bias variance parameter (default: 1.0). Controls the kernel value
        when inputs are zero.
    Amp : float, optional
        Amplitude parameter (default: 1.0). Multiplies the C matrix directly:
        C = Amp * alpha * C_smooth * alpha^T. This affects the kernel non-linearly
        through the sqrt and arccos operations (matching legacy varGP).
        Different from ScaleKernel which scales output linearly.
        Clamped at max 1000.0 by clamp_hyperparameters().
    eps_0x : float, optional
        RF center x-coordinate on [-1, 1] grid (default: 0.0 = center)
    eps_0y : float, optional
        RF center y-coordinate on [-1, 1] grid (default: 0.0 = center)
    beta : float, optional
        RF size parameter (default: 0.1). Smaller = more localized RF.
    rho : float, optional
        Smoothness parameter (default: 0.1). Controls spatial correlation.
    use_mask : bool, optional
        If True, apply pixel masking based on RF parameters.
        Pixels with locality weight α >= 0.001 are included. This reduces C from
        (n_px, n_px) to (n_masked, n_masked), typically ~100x smaller.
        Default: True.
    gradient_mode : str, optional
        How to compute gradients for hyperparameters. Options:
        - 'autograd': PyTorch autograd (default) - automatic differentiation
        - 'vjp': VJP analytical gradients - same speed as autograd, explicit formulas
        - 'jacobian': Jacobian materialization - slow but matches original varGP exactly
        Note: use_analytical_grads parameter was removed (Jan 2025). Use gradient_mode instead.

    Attributes
    ----------
    raw_sigma_0 : Parameter
        Unconstrained parameter for sigma_0
    sigma_0 : Property
        Constrained (positive) sigma_0 value
    raw_Amp : Parameter
        Unconstrained parameter for Amp
    Amp : Property
        Constrained (positive) Amp value
    gradient_mode : str
        Current gradient computation mode
    """

    has_lengthscale = False  # Arc-cosine doesn't have a lengthscale
    MASK_THRESHOLD = 0.001  # Pixels with α >= threshold are included

    # Amp (amplitude) max bound for clamping
    AMP_MAX = 1000.0

    # Bounds for beta (RF size parameter)
    # beta ∈ [0.01, 0.3] → raw ∈ [1.02, 7.82]
    # Upper bound tightened from 1.0 to 0.3 based on active loop investigation
    # (2026-04-10): cell 0 seed 0 random on 108x108 had beta drift 0.12→0.45,
    # causing the C matrix to cover all 11664 pixels (519 MB) and OOM.
    # At beta=0.3, RF sigma = 0.3*sqrt(2) ≈ 0.42 normalized = ~23 pixel sigma
    # on 108x108 (RF diameter ~46 px). Still very generous for any realistic RF.
    BETA_MIN = 0.01
    BETA_MAX = 0.3
    RAW_BETA_MIN = -2 * np.log(2 * BETA_MAX)   # ≈ 0.18
    RAW_BETA_MAX = -2 * np.log(2 * BETA_MIN)   # ≈ 7.82

    # Bounds for rho (smoothness parameter)
    # rho ∈ [0.01, 0.5] → raw ∈ [0.69, 8.52]
    RHO_MIN = 0.01
    RHO_MAX = 0.5
    RAW_RHO_MIN = -np.log(2 * RHO_MAX**2)      # ≈ 0.69
    RAW_RHO_MAX = -np.log(2 * RHO_MIN**2)      # ≈ 8.52

    # Bounds for epsilon (RF center coordinates)
    # Pixel grid is on [-1, 1] × [-1, 1], RF center must stay within image
    EPS_MIN = -1.0
    EPS_MAX = 1.0

    def __init__(self, n_px_side, sigma_0=1.0, Amp=1.0,
                 eps_0x=0.0, eps_0y=0.0, beta=0.1, rho=0.1,
                 use_mask=True, gradient_mode='autograd', **kwargs):
        super().__init__(**kwargs)

        # Register sigma_0 parameter with dummy initial value
        self.register_parameter(
            name='raw_sigma_0',
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        # Positivity via exp transform: raw = log(sigma_0), sigma_0 = exp(raw).
        # Matches paper's parameterization (sigma_0 = exp(sigma_b)).
        self.register_constraint('raw_sigma_0', Positive(transform=torch.exp, inv_transform=torch.log))

        # Now set the actual value via the property (applies inverse transform)
        self.sigma_0 = sigma_0

        # Register Amp parameter (amplitude scaling for C matrix)
        # Amp multiplies C directly: C = Amp * alpha * C_smooth * alpha^T
        # This is different from ScaleKernel which scales the output linearly
        self.register_parameter(
            name='raw_Amp',
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        self.register_constraint('raw_Amp', Positive())
        self.Amp = Amp

        # RF parameters (required)
        if n_px_side is None:
            raise ValueError("n_px_side is required - must specify image dimensions")

        self.n_px_side = n_px_side

        # RF center (unconstrained, range [-1, 1])
        self.register_parameter('eps_0x',
            torch.nn.Parameter(torch.tensor([eps_0x], dtype=torch.float64)))
        self.register_parameter('eps_0y',
            torch.nn.Parameter(torch.tensor([eps_0y], dtype=torch.float64)))

        # Log-space parameters for numerical stability
        # beta = 0.1 → raw_m2log2beta = -2 * log(2 * 0.1) ≈ 3.22
        # rho = 0.1 → raw_mlog2rho2 = -log(2 * 0.1^2) ≈ 3.91
        raw_m2log2beta = -2 * np.log(2 * beta)
        raw_mlog2rho2 = -np.log(2 * rho**2)
        self.register_parameter('raw_m2log2beta',
            torch.nn.Parameter(torch.tensor([raw_m2log2beta], dtype=torch.float64)))
        self.register_parameter('raw_mlog2rho2',
            torch.nn.Parameter(torch.tensor([raw_mlog2rho2], dtype=torch.float64)))

        # Setup pixel coordinate grid
        self._setup_pixel_coords()

        # Masking mode
        self.use_mask = use_mask
        # Cache for mask (computed on first forward pass)
        self._cached_mask = None

        # Warn if masking is disabled (uses full 11664x11664 C matrix)
        if not use_mask:
            warnings.warn(
                f"use_mask=False: Using full {n_px_side**2}x{n_px_side**2} C matrix. "
                "This is memory-intensive (~1GB for 108x108 images). "
                "Set use_mask=True to reduce to ~2500x2500.",
                UserWarning
            )

        # Validate gradient mode
        if gradient_mode not in GRADIENT_MODES:
            raise ValueError(f"gradient_mode must be one of {GRADIENT_MODES}, got '{gradient_mode}'")
        self.gradient_mode = gradient_mode

    @property
    def sigma_0(self):
        """Get the constrained sigma_0 value."""
        return self.raw_sigma_0_constraint.transform(self.raw_sigma_0)

    @sigma_0.setter
    def sigma_0(self, value):
        """Set sigma_0 value."""
        self._set_sigma_0(value)

    def _set_sigma_0(self, value):
        """Set sigma_0 via inverse transform."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_0)
        self.initialize(raw_sigma_0=self.raw_sigma_0_constraint.inverse_transform(value))

    @property
    def Amp(self):
        """Get the constrained Amp value (amplitude scaling for C matrix)."""
        return self.raw_Amp_constraint.transform(self.raw_Amp)

    @Amp.setter
    def Amp(self, value):
        """Set Amp value."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_Amp)
        self.initialize(raw_Amp=self.raw_Amp_constraint.inverse_transform(value))

    @property
    def beta(self):
        """Get the natural beta parameter (RF size).

        Transforms from raw parameterization:
            raw_m2log2beta = -2 * log(2 * beta)
            beta = exp(-raw / 2) / 2

        Returns
        -------
        Tensor
            Natural beta value
        """
        if not hasattr(self, 'raw_m2log2beta'):
            raise AttributeError("beta property only available when n_px_side is set")
        return torch.exp(-self.raw_m2log2beta / 2) * 0.5

    @property
    def rho(self):
        """Get the natural rho parameter (smoothness length scale).

        Transforms from raw parameterization:
            raw_mlog2rho2 = -log(2 * rho^2)
            rho = exp(-raw / 2) / sqrt(2)

        Returns
        -------
        Tensor
            Natural rho value
        """
        if not hasattr(self, 'raw_mlog2rho2'):
            raise AttributeError("rho property only available when n_px_side is set")
        return torch.exp(-self.raw_mlog2rho2 / 2) / np.sqrt(2)

    def params_in_bounds(self):
        """Check whether all kernel hyperparameters are within valid bounds.

        READ-ONLY check. Call at the top of LBFGS closures to reject trial
        steps before expensive computation. Bounds match clamp_hyperparameters().

        Returns
        -------
        bool
            True if all parameters are in bounds.
        """
        with torch.no_grad():
            if hasattr(self, 'raw_sigma_0'):
                if self.sigma_0.item() <= 0:
                    return False
            if hasattr(self, 'raw_Amp'):
                if self.Amp.item() <= 0 or self.Amp.item() > self.AMP_MAX:
                    return False
            if hasattr(self, 'raw_m2log2beta'):
                v = self.raw_m2log2beta.item()
                if v < self.RAW_BETA_MIN or v > self.RAW_BETA_MAX:
                    return False
            if hasattr(self, 'raw_mlog2rho2'):
                v = self.raw_mlog2rho2.item()
                if v < self.RAW_RHO_MIN or v > self.RAW_RHO_MAX:
                    return False
            if hasattr(self, 'eps_0x'):
                if hasattr(self, '_eps_x_min'):
                    # Tight bounds set via set_center_bounds()
                    if self.eps_0x.item() < self._eps_x_min or self.eps_0x.item() > self._eps_x_max:
                        return False
                    if self.eps_0y.item() < self._eps_y_min or self.eps_0y.item() > self._eps_y_max:
                        return False
                else:
                    # Default: full image bounds
                    if self.eps_0x.item() < self.EPS_MIN or self.eps_0x.item() > self.EPS_MAX:
                        return False
                    if self.eps_0y.item() < self.EPS_MIN or self.eps_0y.item() > self.EPS_MAX:
                        return False
        return True

    def clamp_hyperparameters(self):
        """Clamp hyperparameters to valid bounds (projected gradient descent).

        Call this after optimizer.step() to enforce parameter bounds.
        Emits a warning listing which parameters were out of bounds,
        since this should not happen if the LBFGS NaN guard is working.

        Bounds:
            Amp ∈ (0, 1000] (via raw parameter)
            beta ∈ [0.01, 1.0] (via raw parameter)
            rho ∈ [0.01, 0.5] (via raw parameter)
            eps_0x, eps_0y ∈ [-1.0, 1.0] (direct)
        """
        with torch.no_grad():
            violated = []

            if hasattr(self, 'raw_Amp'):
                max_raw_Amp = self.raw_Amp_constraint.inverse_transform(
                    torch.tensor(self.AMP_MAX, device=self.raw_Amp.device, dtype=self.raw_Amp.dtype)
                )
                if self.raw_Amp.item() > max_raw_Amp.item():
                    violated.append(f"Amp={self.Amp.item():.4g} > {self.AMP_MAX}")
                self.raw_Amp.clamp_(max=max_raw_Amp.item())

            if hasattr(self, 'raw_m2log2beta'):
                v = self.raw_m2log2beta.item()
                if v < self.RAW_BETA_MIN or v > self.RAW_BETA_MAX:
                    violated.append(f"raw_m2log2beta={v:.4g} outside [{self.RAW_BETA_MIN:.2f}, {self.RAW_BETA_MAX:.2f}]")
                self.raw_m2log2beta.clamp_(self.RAW_BETA_MIN, self.RAW_BETA_MAX)

            if hasattr(self, 'raw_mlog2rho2'):
                v = self.raw_mlog2rho2.item()
                if v < self.RAW_RHO_MIN or v > self.RAW_RHO_MAX:
                    violated.append(f"raw_mlog2rho2={v:.4g} outside [{self.RAW_RHO_MIN:.2f}, {self.RAW_RHO_MAX:.2f}]")
                self.raw_mlog2rho2.clamp_(self.RAW_RHO_MIN, self.RAW_RHO_MAX)

            if hasattr(self, 'eps_0x'):
                if hasattr(self, '_eps_x_min'):
                    # Tight bounds set via set_center_bounds()
                    if self.eps_0x.item() < self._eps_x_min or self.eps_0x.item() > self._eps_x_max:
                        violated.append(f"eps_0x={self.eps_0x.item():.4g} outside [{self._eps_x_min:.3f}, {self._eps_x_max:.3f}]")
                    if self.eps_0y.item() < self._eps_y_min or self.eps_0y.item() > self._eps_y_max:
                        violated.append(f"eps_0y={self.eps_0y.item():.4g} outside [{self._eps_y_min:.3f}, {self._eps_y_max:.3f}]")
                    self.eps_0x.clamp_(self._eps_x_min, self._eps_x_max)
                    self.eps_0y.clamp_(self._eps_y_min, self._eps_y_max)
                else:
                    # Default: full image bounds
                    if self.eps_0x.item() < self.EPS_MIN or self.eps_0x.item() > self.EPS_MAX:
                        violated.append(f"eps_0x={self.eps_0x.item():.4g}")
                    if self.eps_0y.item() < self.EPS_MIN or self.eps_0y.item() > self.EPS_MAX:
                        violated.append(f"eps_0y={self.eps_0y.item():.4g}")
                    self.eps_0x.clamp_(self.EPS_MIN, self.EPS_MAX)
                    self.eps_0y.clamp_(self.EPS_MIN, self.EPS_MAX)

            if violated:
                warnings.warn(
                    f"clamp_hyperparameters: parameters escaped bounds — {', '.join(violated)}. "
                    f"This suggests the optimizer took a step the NaN guard did not catch."
                )

    def set_center_bounds(self, center_x, center_y, radius):
        """EXPERIMENTAL: Constrain RF center to stay within radius of initial estimate.

        Sets per-axis bounds used by params_in_bounds() and clamp_hyperparameters().
        Bounds are clamped to [EPS_MIN, EPS_MAX] so they never exceed the image.

        Args:
            center_x: Initial RF center x (normalized coords)
            center_y: Initial RF center y (normalized coords)
            radius: Allowed deviation from center (normalized coords)
        """
        self._eps_x_min = max(self.EPS_MIN, center_x - radius)
        self._eps_x_max = min(self.EPS_MAX, center_x + radius)
        self._eps_y_min = max(self.EPS_MIN, center_y - radius)
        self._eps_y_max = min(self.EPS_MAX, center_y + radius)
        print(f"  RF center bounds set: x=[{self._eps_x_min:.3f}, {self._eps_x_max:.3f}], "
              f"y=[{self._eps_y_min:.3f}, {self._eps_y_max:.3f}] (radius={radius:.3f})")

    def _setup_pixel_coords(self):
        """Setup normalized pixel coordinate grid on [-1, 1] × [-1, 1].

        Creates buffers xcord and ycord containing flattened pixel coordinates.
        These are registered as buffers (not parameters) since they're fixed.
        """
        ycord, xcord = torch.meshgrid(
            torch.linspace(-1, 1, self.n_px_side, dtype=torch.float64),
            torch.linspace(-1, 1, self.n_px_side, dtype=torch.float64),
            indexing='ij'
        )
        self.register_buffer('xcord', xcord.flatten())
        self.register_buffer('ycord', ycord.flatten())

    def compute_mask(self):
        """Compute pixel mask based on current RF parameters.

        Mask is computed with DETACHED parameters (structural stability).
        Pixels with locality weight α >= MASK_THRESHOLD are included.

        Returns
        -------
        mask : Tensor, shape (n_px,), dtype=bool
            Boolean mask for active pixels
        """
        # Ensure coords are on same device as parameters
        xcord = self.xcord.to(self.eps_0x.device)
        ycord = self.ycord.to(self.eps_0y.device)

        # Compute mask with DETACHED theta (non-differentiable)
        with torch.no_grad():
            beta_detached = torch.exp(self.raw_m2log2beta.detach())
            dist_sq = (xcord - self.eps_0x.detach())**2 + (ycord - self.eps_0y.detach())**2
            logalpha = -beta_detached * dist_sq
            alpha = torch.exp(logalpha)
            mask = alpha >= self.MASK_THRESHOLD

        return mask

    def _compute_C_matrix(self, apply_mask=False):
        """Compute structured covariance matrix C encoding RF properties.

        C = Amp · α_local[:, None] · C_smooth · α_local[None, :]

        where:
            Amp = amplitude scaling (matches legacy varGP, NOT same as ScaleKernel)
            α_local[i] = exp(-β · ||ξᵢ - ξ₀||²)
            C_smooth[i,j] = exp(-ρ² · ||ξᵢ - ξⱼ||²)

        Note: Amp is multiplied INTO C, affecting the kernel non-linearly through
        the sqrt and arccos operations. This is different from ScaleKernel which
        scales the output linearly.

        Parameters
        ----------
        apply_mask : bool
            If True, compute C only for masked pixels (smaller matrix).
            If False, compute full C matrix.

        Returns
        -------
        C : Tensor
            Structured covariance matrix.
            Shape (n_masked, n_masked) if apply_mask=True, else (n_px, n_px).
        mask : Tensor or None
            Boolean mask (n_px,) if apply_mask=True, else None.
        """
        # Transform log-space parameters
        beta = torch.exp(self.raw_m2log2beta)
        rho2 = torch.exp(self.raw_mlog2rho2)

        # Ensure coords are on same device as parameters
        xcord = self.xcord.to(self.eps_0x.device)
        ycord = self.ycord.to(self.eps_0y.device)

        # Compute mask if needed
        mask = None
        if apply_mask:
            mask = self.compute_mask()
            n_total = mask.numel()
            n_masked = mask.sum().item()
            if n_masked > 0.5 * n_total:
                beta_nat = self.beta.item()
                warnings.warn(
                    f"Mask covers {n_masked}/{n_total} pixels ({100*n_masked/n_total:.0f}%). "
                    f"beta={beta_nat:.4f} is driving a very wide RF. "
                    f"C matrix will be {n_masked}x{n_masked} "
                    f"({n_masked**2 * 4 / 1024**2:.0f} MB float32). "
                    f"Consider whether beta is drifting pathologically.",
                    stacklevel=3,
                )
            xcord = xcord[mask]
            ycord = ycord[mask]

        # Locality weights: distance from RF center
        dist_sq_center = (xcord - self.eps_0x)**2 + (ycord - self.eps_0y)**2
        logalpha = -beta * dist_sq_center
        alpha = torch.exp(logalpha)  # (n_px,) or (n_masked,)

        # Smoothness kernel: pairwise pixel distances
        dx = xcord[:, None] - xcord[None, :]
        dy = ycord[:, None] - ycord[None, :]
        dist_sq_pairwise = dx**2 + dy**2
        C_smooth = torch.exp(-rho2 * dist_sq_pairwise)

        # Full C matrix: Amp * outer product of alpha weighted by C_smooth
        # This matches legacy varGP: C = theta['Amp'] * alpha * C_smooth * alpha^T
        C = self.Amp * alpha[:, None] * C_smooth * alpha[None, :]

        # Symmetrize for numerical stability
        C = (C + C.T) / 2

        return C, mask

    def forward(self, x1, x2, diag=False, **params):
        """Compute the arc-cosine kernel matrix.

        Parameters
        ----------
        x1 : Tensor, shape (..., n1, n_features)
            First input batch. When use_mask=True, inputs should be FULL images
            (n_features = n_px_side²) and masking is applied internally.
        x2 : Tensor, shape (..., n2, n_features)
            Second input batch
        diag : bool
            If True, return only the diagonal of the kernel matrix

        Returns
        -------
        Tensor
            Kernel matrix of shape (..., n1, n2) or (..., n1,) if diag=True
        """
        # Use analytical gradients path if enabled (VJP or Jacobian)
        if self.gradient_mode in ('vjp', 'jacobian') and self.n_px_side is not None:
            if self.gradient_mode == 'vjp':
                GradFunction = _get_vjp_implementation()
            else:  # jacobian
                GradFunction = _get_jacobian_implementation()

            K = GradFunction.apply(
                x1, x2,
                self.sigma_0,
                self.Amp,
                self.eps_0x, self.eps_0y,
                self.raw_m2log2beta, self.raw_mlog2rho2,
                self.n_px_side, self.use_mask, diag
            )
            return K

        sigma_0_sq = self.sigma_0 ** 2

        # Compute C from RF parameters
        C, mask = self._compute_C_matrix(apply_mask=self.use_mask)
        # Cache mask for external access
        if mask is not None:
            self._cached_mask = mask

        # Apply mask to inputs if using masking
        if mask is not None:
            x1 = x1[..., mask]
            x2 = x2[..., mask]

        # Compute quadratic form xᵀCx
        C = C.to(x1.device, x1.dtype)
        CX1 = x1 @ C  # (..., n1, n_features)
        V1 = (CX1 * x1).sum(dim=-1) + sigma_0_sq  # (..., n1)

        if diag:
            # Diagonal case: K(x_i, x_i)
            # When x1 = x2, cos(θ) = 1, θ = 0, J(0) = π
            # K = M * π / π = M = √(v_x * v_x) = v_x
            return V1

        # Full matrix case
        CX2 = x2 @ C  # (..., n2, n_features)
        V2 = (CX2 * x2).sum(dim=-1) + sigma_0_sq  # (..., n2)

        # Cross-term: xᵀCx'
        # Shape: (..., n1, n2)
        C12 = torch.matmul(CX1, x2.transpose(-2, -1)) + sigma_0_sq

        # Magnitude: M = √(v_x · v_x')
        # Broadcast V1 (..., n1) and V2 (..., n2) to (..., n1, n2)
        M = torch.sqrt(V1.unsqueeze(-1) * V2.unsqueeze(-2))

        # Normalized inner product
        # Clamp for numerical stability (arccos domain is [-1, 1])
        eps = 1e-7
        cos_theta = torch.clamp(C12 / M, -1.0 + eps, 1.0 - eps)

        # Angle and sine
        theta_angle = torch.arccos(cos_theta)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=eps))

        # Angular term: J(θ) = sin(θ) + (π - θ)cos(θ)
        J = sin_theta + (torch.pi - theta_angle) * cos_theta

        # Kernel: K = M · J / π
        K = M * J / torch.pi

        return K


class ArcCosineKernelNormalized(ArcCosineKernel):
    """Normalized arc-cosine kernel with full RF structure.

    K_bar(x, x') = J(theta) / pi
    K_bar(x, x) = 1.0  (constant diagonal)

    The magnitude factor M = sqrt(v_x * v_x') is dropped from the
    unnormalized kernel K = M * J / pi. This eliminates the prior
    variance's dependence on input norm (v_x = x^T C x + sigma_0^2),
    giving constant prior variance everywhere.

    All RF structure (C matrix, masking, eigenspace projection) is
    inherited from ArcCosineKernel. Only autograd gradient mode is
    supported (VJP/Jacobian would need separate derivations).

    Parameters: Same as ArcCosineKernel. gradient_mode is forced to 'autograd'.
    """

    def __init__(self, n_px_side, sigma_0=1.0, Amp=1.0,
                 eps_0x=0.0, eps_0y=0.0, beta=0.1, rho=0.1,
                 use_mask=True, gradient_mode='autograd', **kwargs):
        if gradient_mode != 'autograd':
            warnings.warn(
                f"ArcCosineKernelNormalized only supports autograd gradient mode. "
                f"Ignoring gradient_mode='{gradient_mode}'."
            )
        super().__init__(
            n_px_side=n_px_side, sigma_0=sigma_0, Amp=Amp,
            eps_0x=eps_0x, eps_0y=eps_0y, beta=beta, rho=rho,
            use_mask=use_mask, gradient_mode='autograd', **kwargs
        )

    def forward(self, x1, x2, diag=False, **params):
        """Compute normalized arc-cosine kernel.

        Returns J(theta)/pi for off-diagonal, 1.0 for diagonal.
        Identical computation to parent except M is NOT multiplied
        into the result and diagonal returns ones.
        """
        sigma_0_sq = self.sigma_0 ** 2

        C, mask = self._compute_C_matrix(apply_mask=self.use_mask)
        if mask is not None:
            self._cached_mask = mask

        if mask is not None:
            x1 = x1[..., mask]
            x2 = x2[..., mask]

        C = C.to(x1.device, x1.dtype)
        CX1 = x1 @ C
        V1 = (CX1 * x1).sum(dim=-1) + sigma_0_sq

        if diag:
            # Normalized: K_bar(x,x) = J(0)/pi = pi/pi = 1
            return torch.ones(V1.shape, dtype=V1.dtype, device=V1.device)

        CX2 = x2 @ C
        V2 = (CX2 * x2).sum(dim=-1) + sigma_0_sq

        C12 = torch.matmul(CX1, x2.transpose(-2, -1)) + sigma_0_sq

        M = torch.sqrt(V1.unsqueeze(-1) * V2.unsqueeze(-2))

        eps = 1e-7
        cos_theta = torch.clamp(C12 / M, -1.0 + eps, 1.0 - eps)

        theta_angle = torch.arccos(cos_theta)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=eps))

        J = sin_theta + (torch.pi - theta_angle) * cos_theta

        # Normalized: drop M (it cancels in K/sqrt(K(x,x)*K(x',x')))
        return J / torch.pi


class ArcSineKernel(ArcCosineKernel):
    """Arc-sine kernel (Williams 1998) with full RF structure.

    K_sat(x, x') = (2/pi) * arcsin( (x^T C x' + sigma_0^2)
                                     / sqrt((1 + v_x)(1 + v_x')) )

    where v_x = x^T C x + sigma_0^2. Derived from an infinite-width
    single-hidden-layer network with erf activation (same depth as
    arc-cosine, different nonlinearity).

    K_sat(x,x) saturates smoothly at 1 for large v_x, preventing the
    quadratic growth of the arc-cosine kernel. The "+1" in the denominator
    is intrinsic to the erf-network derivation, not a design choice.

    All RF structure (C matrix, masking, parameter bounds) is inherited
    from ArcCosineKernel. Only autograd gradient mode is supported.

    TEMPORARY: Subclasses ArcCosineKernel for expedience. If this kernel
    proves useful, the shared C-matrix infrastructure should be factored
    into a dedicated base class.

    Parameters: Same as ArcCosineKernel. gradient_mode forced to 'autograd'.
    """

    def __init__(self, n_px_side, sigma_0=1.0, Amp=1.0,
                 eps_0x=0.0, eps_0y=0.0, beta=0.1, rho=0.1,
                 use_mask=True, gradient_mode='autograd', **kwargs):
        if gradient_mode != 'autograd':
            warnings.warn(
                f"ArcSineKernel only supports autograd gradient mode. "
                f"Ignoring gradient_mode='{gradient_mode}'."
            )
        super().__init__(
            n_px_side=n_px_side, sigma_0=sigma_0, Amp=Amp,
            eps_0x=eps_0x, eps_0y=eps_0y, beta=beta, rho=rho,
            use_mask=use_mask, gradient_mode='autograd', **kwargs
        )

    def forward(self, x1, x2, diag=False, **params):
        """Compute the arc-sine kernel matrix.

        K_sat(x, x') = (2/pi) * arcsin( (x^T C x' + sigma_0^2)
                                         / sqrt((1 + v_x)(1 + v_x')) )

        Diagonal: K_sat(x,x) = (2/pi) * arcsin(v_x / (1 + v_x))
        This is NOT constant (unlike normalized kernel) — it varies
        with input magnitude but saturates at 1.
        """
        sigma_0_sq = self.sigma_0 ** 2

        C, mask = self._compute_C_matrix(apply_mask=self.use_mask)
        if mask is not None:
            self._cached_mask = mask
            x1 = x1[..., mask]
            x2 = x2[..., mask]

        C = C.to(x1.device, x1.dtype)
        CX1 = x1 @ C
        V1 = (CX1 * x1).sum(dim=-1) + sigma_0_sq  # v_x

        if diag:
            # K_sat(x,x) = (2/pi) * arcsin(v_x / (1 + v_x))
            arg = V1 / (1.0 + V1)
            return (2.0 / torch.pi) * torch.arcsin(arg)

        CX2 = x2 @ C
        V2 = (CX2 * x2).sum(dim=-1) + sigma_0_sq

        C12 = torch.matmul(CX1, x2.transpose(-2, -1)) + sigma_0_sq
        denom = torch.sqrt((1.0 + V1).unsqueeze(-1) * (1.0 + V2).unsqueeze(-2))

        eps = 1e-7
        arg = torch.clamp(C12 / denom, -1.0 + eps, 1.0 - eps)
        return (2.0 / torch.pi) * torch.arcsin(arg)


class LocalRBFKernel(ArcCosineKernel):
    """RBF kernel with receptive field structure (C matrix) and lengthscale.

    k(x, x') = exp(-(x - x')^T C_base (x - x') / (2 l^2))

    where C_base = diag(alpha) @ C_smooth @ diag(alpha) is the RF
    structure matrix (same as ArcCosineKernel but WITHOUT the Amp factor),
    and l is a learnable lengthscale stored in log-space.

    This cleanly separates:
    - C_base: RF structure (center, size, smoothness) — from beta, rho, eps
    - l: distance sensitivity (how different images must be before K drops)

    This is a STATIONARY kernel: depends only on (x - x'), not absolute
    values. Consequences:
    - k(x, x) = 1 for all x (constant prior variance)
    - No sigma_0 effect (bias cancels in the difference)
    - All "drive" must come from likelihood (A, lambda0)

    Only autograd gradient mode supported.

    NOTE: sigma_0 and Amp are inherited from ArcCosineKernel but NOT used
    in forward(). They exist for inheritance convenience. C_base excludes
    Amp; the lengthscale parameter replaces its role.

    TEMPORARY: Subclasses ArcCosineKernel for expedience (same pattern as
    ArcSineKernel). If useful, shared C-matrix infrastructure should be
    factored into a dedicated base class.

    Parameters
    ----------
    lengthscale : float
        Initial lengthscale value (default: 100.0). Stored in log-space
        as raw_log_lengthscale. Controls kernel sharpness: larger l means
        more tolerant (flatter kernel), smaller l means more sensitive.
    Other parameters: Same as ArcCosineKernel. sigma_0 and Amp have no effect.
    gradient_mode forced to 'autograd'.
    """

    # Lengthscale bounds
    LENGTHSCALE_MIN = 0.1
    LENGTHSCALE_MAX = 100000.0
    RAW_LS_MIN = np.log(LENGTHSCALE_MIN)   # approx -2.30
    RAW_LS_MAX = np.log(LENGTHSCALE_MAX)   # approx 11.51

    def __init__(self, n_px_side, sigma_0, Amp,
                 eps_0x, eps_0y, beta, rho,
                 lengthscale,
                 use_mask=True, gradient_mode='autograd', **kwargs):
        if gradient_mode != 'autograd':
            warnings.warn(
                f"LocalRBFKernel only supports autograd gradient mode. "
                f"Ignoring gradient_mode='{gradient_mode}'."
            )
        super().__init__(
            n_px_side=n_px_side, sigma_0=sigma_0, Amp=Amp,
            eps_0x=eps_0x, eps_0y=eps_0y, beta=beta, rho=rho,
            use_mask=use_mask, gradient_mode='autograd', **kwargs
        )

        # Lengthscale in log-space (RBF-specific parameter)
        raw_log_ls = np.log(lengthscale)
        self.register_parameter('raw_log_lengthscale',
            torch.nn.Parameter(torch.tensor([raw_log_ls], dtype=torch.float64)))

    @property
    def lengthscale(self):
        """Get the lengthscale l = exp(raw_log_lengthscale)."""
        return torch.exp(self.raw_log_lengthscale)

    def _compute_C_matrix(self, apply_mask=False):
        """Compute C_base WITHOUT Amp — lengthscale handles distance scaling.

        C_base = diag(alpha) @ C_smooth @ diag(alpha)

        This is the same RF structure as ArcCosineKernel._compute_C_matrix()
        but without the Amp multiplication. The lengthscale parameter in
        forward() replaces Amp's role for the RBF kernel.
        """
        beta = torch.exp(self.raw_m2log2beta)
        rho2 = torch.exp(self.raw_mlog2rho2)

        xcord = self.xcord.to(self.eps_0x.device)
        ycord = self.ycord.to(self.eps_0y.device)

        mask = None
        if apply_mask:
            mask = self.compute_mask()
            xcord = xcord[mask]
            ycord = ycord[mask]

        # Locality weights: distance from RF center
        dist_sq_center = (xcord - self.eps_0x)**2 + (ycord - self.eps_0y)**2
        alpha = torch.exp(-beta * dist_sq_center)

        # Smoothness kernel: pairwise pixel distances
        dx = xcord[:, None] - xcord[None, :]
        dy = ycord[:, None] - ycord[None, :]
        C_smooth = torch.exp(-rho2 * (dx**2 + dy**2))

        # C_base: NO Amp multiplication — lengthscale handles scaling
        C = alpha[:, None] * C_smooth * alpha[None, :]
        C = (C + C.T) / 2

        return C, mask

    def forward(self, x1, x2, diag=False, **params):
        """Compute the local RBF kernel matrix.

        k(x, x') = exp(-(x - x')^T C_base (x - x') / (2 l^2))

        Expanded quadratic form:
            (x-y)^T C (x-y) = x^T C x - 2 x^T C y + y^T C y

        Diagonal: k(x, x) = exp(0) = 1 for all x.
        """
        C, mask = self._compute_C_matrix(apply_mask=self.use_mask)
        if mask is not None:
            self._cached_mask = mask
            x1 = x1[..., mask]
            x2 = x2[..., mask]

        C = C.to(x1.device, x1.dtype)

        if diag:
            return torch.ones(x1.shape[:-1], dtype=x1.dtype, device=x1.device)

        # Quadratic expansion: (x-y)^T C (x-y) = x^T C x - 2 x^T C y + y^T C y
        X1_C = x1 @ C                                          # (..., n1, n_masked)
        X2_C = x2 @ C                                          # (..., n2, n_masked)
        self_x1 = (x1 * X1_C).sum(dim=-1)                      # (..., n1)
        self_x2 = (x2 * X2_C).sum(dim=-1)                      # (..., n2)
        cross = torch.matmul(X1_C, x2.transpose(-2, -1))       # (..., n1, n2)

        dist_sq = self_x1.unsqueeze(-1) - 2 * cross + self_x2.unsqueeze(-2)
        dist_sq = torch.clamp(dist_sq, min=0.0)  # prevent negative from float errors

        ls_sq = self.lengthscale ** 2
        return torch.exp(-0.5 * dist_sq / ls_sq)

    def params_in_bounds(self):
        """Check all hyperparameters including lengthscale."""
        if not super().params_in_bounds():
            return False
        with torch.no_grad():
            v = self.raw_log_lengthscale.item()
            if v < self.RAW_LS_MIN or v > self.RAW_LS_MAX:
                return False
        return True

    def clamp_hyperparameters(self):
        """Clamp hyperparameters including lengthscale."""
        super().clamp_hyperparameters()
        with torch.no_grad():
            v = self.raw_log_lengthscale.item()
            if v < self.RAW_LS_MIN or v > self.RAW_LS_MAX:
                warnings.warn(
                    f"clamp_hyperparameters: raw_log_lengthscale={v:.4g} "
                    f"outside [{self.RAW_LS_MIN:.2f}, {self.RAW_LS_MAX:.2f}]"
                )
            self.raw_log_lengthscale.clamp_(self.RAW_LS_MIN, self.RAW_LS_MAX)


class SimpleArcCosineKernel(Kernel):
    """Arc-cosine kernel for low-dimensional playground inputs (NOT images).

    For simple 2D/3D coordinate inputs where C = identity. No RF structure,
    no pixel masking, no Amp parameter (Amp scales C, which doesn't apply
    when C = I; use ScaleKernel for linear output scaling if needed).

    K(x, x') = (1/pi) * M * J(theta)

    where:
        v_x     = x^T x + sigma_0^2
        v_x'    = x'^T x' + sigma_0^2
        M       = sqrt(v_x * v_x')
        cos(theta) = (x^T x' + sigma_0^2) / M
        J(theta) = sin(theta) + (pi - theta) * cos(theta)

    Diagonal: K(x, x) = v_x = ||x||^2 + sigma_0^2

    Parameters
    ----------
    sigma_0 : float
        Bias variance parameter (default: 1.0).
    """

    has_lengthscale = False

    def __init__(self, sigma_0=1.0, **kwargs):
        super().__init__(**kwargs)

        self.register_parameter(
            name='raw_sigma_0',
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        self.register_constraint('raw_sigma_0', Positive(transform=torch.exp, inv_transform=torch.log))
        self.sigma_0 = sigma_0

    @property
    def sigma_0(self):
        return self.raw_sigma_0_constraint.transform(self.raw_sigma_0)

    @sigma_0.setter
    def sigma_0(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_0)
        self.initialize(raw_sigma_0=self.raw_sigma_0_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        sigma_0_sq = self.sigma_0 ** 2

        # v_x = x^T x + sigma_0^2  (C = I)
        V1 = (x1 * x1).sum(dim=-1) + sigma_0_sq  # (..., n1)

        if diag:
            # K(x, x) = v_x (since theta=0, J(0)=pi, K = M*pi/pi = M = v_x)
            return V1

        V2 = (x2 * x2).sum(dim=-1) + sigma_0_sq  # (..., n2)

        # Cross-term: x^T x' + sigma_0^2
        C12 = torch.matmul(x1, x2.transpose(-2, -1)) + sigma_0_sq  # (..., n1, n2)

        # M = sqrt(v_x * v_x')
        M = torch.sqrt(V1.unsqueeze(-1) * V2.unsqueeze(-2))  # (..., n1, n2)

        # Normalized inner product
        eps = 1e-7
        cos_theta = torch.clamp(C12 / M, -1.0 + eps, 1.0 - eps)

        theta_angle = torch.arccos(cos_theta)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=eps))

        # J(theta) = sin(theta) + (pi - theta) * cos(theta)
        J = sin_theta + (torch.pi - theta_angle) * cos_theta

        return M * J / torch.pi


class SimpleArcCosineNormalizedKernel(Kernel):
    """Normalized arc-cosine kernel for low-dimensional playground inputs.

    K_bar(x, x') = K(x, x') / sqrt(K(x,x) * K(x',x'))
                  = (1/pi) * J(theta)

    The magnitude factor M = sqrt(v_x * v_x') cancels exactly in the
    normalization, leaving only the angular term. This gives constant
    prior variance K_bar(x, x) = 1 for all x, eliminating the norm-scaling
    incentive that causes utility optimization to diverge to domain corners.

    sigma_0 still appears inside theta:
        cos(theta) = (x^T x' + sigma_0^2) / sqrt((||x||^2 + sigma_0^2)(||x'||^2 + sigma_0^2))

    Parameters
    ----------
    sigma_0 : float
        Bias variance parameter (default: 1.0).
    """

    has_lengthscale = False

    def __init__(self, sigma_0=1.0, **kwargs):
        super().__init__(**kwargs)

        self.register_parameter(
            name='raw_sigma_0',
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        self.register_constraint('raw_sigma_0', Positive(transform=torch.exp, inv_transform=torch.log))
        self.sigma_0 = sigma_0

    @property
    def sigma_0(self):
        return self.raw_sigma_0_constraint.transform(self.raw_sigma_0)

    @sigma_0.setter
    def sigma_0(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_0)
        self.initialize(raw_sigma_0=self.raw_sigma_0_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        sigma_0_sq = self.sigma_0 ** 2

        # v_x = x^T x + sigma_0^2  (C = I)
        V1 = (x1 * x1).sum(dim=-1) + sigma_0_sq  # (..., n1)

        if diag:
            # K_bar(x, x) = J(0)/pi = pi/pi = 1
            return torch.ones(V1.shape, dtype=V1.dtype, device=V1.device)

        V2 = (x2 * x2).sum(dim=-1) + sigma_0_sq  # (..., n2)

        # Cross-term: x^T x' + sigma_0^2
        C12 = torch.matmul(x1, x2.transpose(-2, -1)) + sigma_0_sq  # (..., n1, n2)

        # M = sqrt(v_x * v_x') — needed for cos_theta, NOT multiplied into output
        M = torch.sqrt(V1.unsqueeze(-1) * V2.unsqueeze(-2))  # (..., n1, n2)

        # Normalized inner product
        eps = 1e-7
        cos_theta = torch.clamp(C12 / M, -1.0 + eps, 1.0 - eps)

        theta_angle = torch.arccos(cos_theta)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=eps))

        # J(theta) = sin(theta) + (pi - theta) * cos(theta)
        J = sin_theta + (torch.pi - theta_angle) * cos_theta

        # Normalized: drop M factor (magnitude cancels in K/sqrt(K*K))
        return J / torch.pi
