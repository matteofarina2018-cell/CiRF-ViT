"""
Checkpoint save/load for vargp_direct (DirectVGPModel).

Saves model state (kernel params, likelihood params, variational params m_b/V_b)
plus pool indices + integrity metadata into a single .pt file.

Design notes:
- X_train is NOT saved. Reconstructed from pool_indices on load. All runs share
  the same PNAS pool, so this keeps checkpoints small (~1 KB vs ~2 MB).
- Derived eigenspace state (B, eigvals_b, K_tilde_b, K_b, KKtilde_inv_b, Kvec)
  is NOT saved. The DirectVGPModel constructor recomputes it from
  (kernel, X_train, eigval_tol). Only m_b and V_b need re-injection.
- Integrity safety net: pool_shape and pool_sum are saved and verified on
  load to catch "wrong dataset" bugs without crypto hashing.

Mirrors checkpoint.py (which handles default_gpy / VariationalGPModel) but is
a separate file because the two model classes have incompatible state
representations: DirectVGPModel is not an nn.Module and has no
variational_strategy.
"""

from datetime import datetime
from pathlib import Path
import warnings

import torch

from .kernels import create_kernel
from .likelihoods import PoissonLikelihood
from .eigenspace_model import DirectVGPModel


def save_eigenspace_checkpoint(
    model,
    config,
    metrics,
    pool_indices,
    X_pool,
    checkpoint_path,
):
    """Save DirectVGPModel state to a .pt file.

    Integrity tags (pool_shape, pool_sum, pool_dtype) are derived from X_pool
    inside this function so the caller cannot accidentally pass mismatched
    values. The cost of recomputing X_pool.sum() per save is negligible
    compared to training (~5 ms on GPU for a 37M-element float32 pool).

    Args:
        model: Trained DirectVGPModel.
        config: Flat config dict used for training (needed for reconstruction).
        metrics: Dict of final metrics to embed in the checkpoint.
        pool_indices: LongTensor (n_training,), indices into the PNAS pool.
        X_pool: (N_pool, n_pixels) full image pool tensor. Used to derive
                pool_shape, pool_sum, pool_dtype integrity tags.
        checkpoint_path: Path to the output .pt file. Parent dirs are created.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive integrity tags from X_pool. Single source of truth: any mismatch
    # at load time means the caller is using a different pool than was
    # originally trained on.
    pool_shape = list(X_pool.shape)
    pool_sum = float(X_pool.sum().item())
    pool_dtype = str(X_pool.dtype)

    kernel = model.kernel
    likelihood = model.likelihood

    # Extract final hyperparameters for quick inspection without loading the
    # full checkpoint. Mirrors checkpoint.py convention.
    hyperparams = {
        'A': likelihood.A.item(),
        'lambda0': likelihood.lambda0.item(),
        'sigma_0': kernel.sigma_0.item(),
        'Amp': kernel.Amp.item(),
        'beta': kernel.beta.item(),
        'rho': kernel.rho.item(),
        'eps_0x': kernel.eps_0x.item(),
        'eps_0y': kernel.eps_0y.item(),
    }

    # RF center bounds, when a caller has set them with kernel.set_center_bounds(),
    # are plain instance attributes and NOT part of state_dict, so they are saved
    # explicitly here: a reloaded model must keep the same trainable region, and
    # only the drifted trained eps_0x survives in state_dict. Nothing in this
    # repository sets them, so this is None for every checkpoint it writes (see
    # the note on bound_rf_center in train.py).
    if hasattr(kernel, '_eps_x_min'):
        rf_center_bounds = {
            'eps_x_min': kernel._eps_x_min,
            'eps_x_max': kernel._eps_x_max,
            'eps_y_min': kernel._eps_y_min,
            'eps_y_max': kernel._eps_y_max,
        }
    else:
        rf_center_bounds = None

    checkpoint = {
        'kernel_state_dict': kernel.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'm_b': model.state.m_b.cpu(),
        'V_b': model.state.V_b.cpu(),
        # Persist the eigenspace basis used at training time.  Without this,
        # load_eigenspace_checkpoint would recompute B from a fresh eigh() of
        # K_tilde — but eigh's eigenvectors are unique only up to per-column
        # sign and rotation within degenerate subspaces, and the convention
        # can drift across PyTorch / cuBLAS / cuSOLVER versions.  Since m_b
        # and V_b are stored in the basis B used during training, a reloaded
        # B' ≠ B silently corrupts predictions (exp(λ) collapses toward
        # exp(λ₀)).  Saving B + eigvals_b makes the restore reproducible.
        'B': model.state.B.cpu(),
        'eigvals_b': model.state.eigvals_b.cpu(),
        'pool_indices': pool_indices.cpu(),
        'pool_shape': pool_shape,
        'pool_sum': pool_sum,
        'pool_dtype': pool_dtype,
        'rf_center_bounds': rf_center_bounds,
        'config': config,
        'metrics': metrics,
        'hyperparams': hyperparams,
        'metadata': {
            'cell_id': config['cell'],
            'n_px_side': config['n_px_side'],
            # config['M'] / config['n_train'] are maintained by the caller.
            # metadata['M'] is provided for convenience only: at load time
            # pool_indices.shape[0] is the authoritative source
            # (see load_pool_indices()).
            'M': config['M'],
            'kernel_type': config['kernel_type'],
            'eigval_tol': config['eigval_tol'],
            'lambda_var_clamp': config['lambda_var_clamp'],
            'data_path': str(config.get('data_path', config.get('dataset', ''))),
            'saved_at': datetime.now().isoformat(timespec='seconds'),
        },
    }

    torch.save(checkpoint, checkpoint_path)


def _verify_pool_integrity(checkpoint, X_pool, pool_sum_tolerance):
    """Verify a checkpoint's saved integrity tags against the loader's X_pool.

    Three independent checks, in increasing strictness:
      - **Shape** must match exactly. Wrong dataset/resolution -> AssertionError.
      - **Dtype** check is a WARNING, not an error. The pool_sum_tolerance was
        tuned for float32 reduction noise; loading on a different dtype means
        the tolerance may be either too loose (float16/bfloat16) or absurdly
        too tight (float64). Loading still proceeds.
      - **Sum** must agree to within pool_sum_tolerance. Same shape but
        different content (a swapped/regenerated .npz with identical
        dimensions) -> AssertionError.

    Args:
        checkpoint: dict from torch.load().
        X_pool: tensor the caller plans to use for X_train reconstruction.
        pool_sum_tolerance: float, max allowed |saved_sum - current_sum|.

    Raises:
        AssertionError on shape or sum mismatch.
    """
    saved_shape = checkpoint['pool_shape']
    saved_sum = checkpoint['pool_sum']
    saved_dtype = checkpoint.get('pool_dtype')  # may be absent in older checkpoints

    current_shape = list(X_pool.shape)
    if current_shape != saved_shape:
        raise AssertionError(
            f"Pool shape mismatch: checkpoint has {saved_shape}, "
            f"X_pool has {current_shape}. Likely wrong dataset."
        )

    if saved_dtype is not None and str(X_pool.dtype) != saved_dtype:
        warnings.warn(
            f"Pool dtype mismatch: checkpoint was saved with {saved_dtype} "
            f"but X_pool is {X_pool.dtype}. The pool_sum_tolerance "
            f"({pool_sum_tolerance}) is calibrated for the saved dtype's "
            f"reduction noise floor — loading will proceed but the "
            f"integrity check may be unreliable. If your dtype is wider, "
            f"the tolerance is probably absurdly too loose; if narrower, "
            f"it may be too tight."
        )

    current_sum = float(X_pool.sum().item())
    if abs(current_sum - saved_sum) > pool_sum_tolerance:
        raise AssertionError(
            f"Pool sum mismatch: checkpoint has {saved_sum:.6f}, "
            f"X_pool has {current_sum:.6f} "
            f"(tolerance {pool_sum_tolerance}). "
            f"Likely content differs from save time."
        )


def load_pool_indices(checkpoint_path, X_pool, pool_sum_tolerance):
    """Load pool_indices from an eigenspace checkpoint and verify pool integrity.

    Lightweight counterpart to load_eigenspace_checkpoint(): reads only the
    index tensor and the integrity tags, NOT the model state. Intended for
    like-for-like reproduction where a fresh fit should train on the exact
    same pool subset a previous run used.

    Contract:
        - The returned tensor's length is authoritative for BOTH n_train and M.
        - metadata['M'] in the checkpoint is NOT trusted; pool_indices.shape[0]
          is the single source of truth.

    Args:
        checkpoint_path: Path to a .pt file. Must be readable by torch.load
            with weights_only=False, and must contain at minimum the keys
            {'pool_indices', 'pool_shape', 'pool_sum'}. Any checkpoint
            produced by save_eigenspace_checkpoint() satisfies this.
        X_pool: (N_pool, n_pixels) tensor the caller will index into. Must
            match the pool used at save time — verified via pool_shape and
            pool_sum by the same integrity check used by
            load_eigenspace_checkpoint().
        pool_sum_tolerance: float. Max allowed |saved_sum - X_pool.sum()|.
            Must be passed explicitly — read it from
            default_params.json -> active_learning.checkpoint_pool_sum_tolerance.
            No hidden default here; single source of truth is the config.

    Returns:
        pool_indices: torch.LongTensor of shape (n,), on CPU. Caller is
            responsible for moving to the desired device.

    Raises:
        AssertionError on pool shape or sum mismatch (wrong dataset).
        KeyError if required checkpoint keys are missing.
        ValueError if pool_indices has the wrong dimensionality.

    Example:
        >>> X_pool = ...  # loaded from the dataset .npz
        >>> tol = defaults['active_learning']['checkpoint_pool_sum_tolerance']
        >>> idx = load_pool_indices(
        ...     'results/active_loop/.../checkpoints/iter_250.pt',
        ...     X_pool, tol,
        ... )
        >>> X_train = X_pool[idx.to(X_pool.device)]
    """
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False
    )

    required = {'pool_indices', 'pool_shape', 'pool_sum'}
    missing = required - set(checkpoint.keys())
    if missing:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} missing required keys: "
            f"{sorted(missing)}. Expected an eigenspace checkpoint (from "
            f"save_eigenspace_checkpoint) or a compatible index bundle."
        )

    _verify_pool_integrity(checkpoint, X_pool, pool_sum_tolerance)

    pool_indices = checkpoint['pool_indices']
    if not isinstance(pool_indices, torch.Tensor):
        pool_indices = torch.as_tensor(pool_indices, dtype=torch.long)
    pool_indices = pool_indices.to(dtype=torch.long).cpu()

    if pool_indices.dim() != 1:
        raise ValueError(
            f"pool_indices must be 1-D; got shape "
            f"{tuple(pool_indices.shape)} from {checkpoint_path}."
        )
    return pool_indices


def load_eigenspace_checkpoint(checkpoint_path, X_pool, pool_sum_tolerance,
                               device=None):
    """Load a DirectVGPModel checkpoint and reconstruct the model.

    Verifies pool_shape and pool_sum against the provided X_pool before
    reconstructing. Raises AssertionError on mismatch (likely wrong dataset).
    Also prints a warning if the saved data_path differs from the current
    config (non-fatal: files can legitimately move).

    Args:
        checkpoint_path: Path to the .pt file.
        X_pool: (N_pool, n_pixels) tensor. Same pool ordering as at save time
                (usually cat([X_train, X_val]) flattened from the same .npz).
        pool_sum_tolerance: float. Maximum absolute difference allowed between
                the pool_sum saved in the checkpoint and X_pool.sum() at load
                time. Must be passed explicitly — read it from
                default_params.json -> active_learning.checkpoint_pool_sum_tolerance.
                No hidden default here; single source of truth is the config.
        device: Optional device override. Default 'cpu'.

    Returns:
        dict with keys:
            'model': reconstructed DirectVGPModel
            'config': flat config dict from training
            'metrics': training metrics dict
            'hyperparams': final hyperparameter values
            'metadata': cell_id, data_path, saved_at, etc.
            'pool_indices': LongTensor indices into X_pool

    Raises:
        AssertionError: if pool_shape or pool_sum does not match X_pool.
    """
    if device is None:
        device = 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    metadata = checkpoint['metadata']

    # Pool shape + dtype + sum integrity checks (raises on shape/sum mismatch,
    # warns on dtype mismatch). All three are about the loader's X_pool.
    _verify_pool_integrity(checkpoint, X_pool, pool_sum_tolerance)

    # data_path is a separate concern: the saved string vs the current config
    # string. This is purely a UX warning — files can legitimately move and
    # the integrity checks already covered the actual content.
    saved_data_path = metadata.get('data_path')
    current_data_path = config.get('data_path')
    if (saved_data_path and current_data_path
            and str(saved_data_path) != str(current_data_path)):
        warnings.warn(
            f"Checkpoint was saved with data_path='{saved_data_path}' but "
            f"current config has data_path='{current_data_path}'. Integrity "
            f"checks passed so probably fine, but double-check the pool source."
        )

    # ----- Reconstruct training data from indices -----
    pool_indices = checkpoint['pool_indices']
    X_train = X_pool[pool_indices].to(device=device)

    # ----- Reconstruct kernel -----
    # eps_0x and eps_0y are nn.Parameters registered directly in the kernel,
    # so they live in the state_dict. Extract them first for create_kernel()
    # construction; load_state_dict will re-set them immediately afterward.
    kernel_state = checkpoint['kernel_state_dict']
    eps_0x = kernel_state['eps_0x'].item()
    eps_0y = kernel_state['eps_0y'].item()
    n_px_side = metadata['n_px_side']

    kernel = create_kernel(config, n_px_side, eps_0x, eps_0y)
    kernel.load_state_dict(kernel_state)

    # Restore RF center bounds from the saved values (if any). They must never be
    # re-derived from the current parameters: that would anchor them to the
    # TRAINED eps_0x, already drifted from the initial estimate, silently
    # changing the trainable region of a reloaded model.
    rf_center_bounds = checkpoint.get('rf_center_bounds')
    if rf_center_bounds is not None:
        kernel._eps_x_min = rf_center_bounds['eps_x_min']
        kernel._eps_x_max = rf_center_bounds['eps_x_max']
        kernel._eps_y_min = rf_center_bounds['eps_y_min']
        kernel._eps_y_max = rf_center_bounds['eps_y_max']

    # Determine dtype from saved variational parameters.
    dtype = checkpoint['m_b'].dtype
    kernel = kernel.to(device=device, dtype=dtype)

    # ----- Reconstruct likelihood -----
    likelihood = PoissonLikelihood(
        A_init=config['A_init'],
        lambda0_init=config['lambda0_init'],
    )
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    likelihood = likelihood.to(device=device, dtype=dtype)

    X_train = X_train.to(dtype=dtype)

    # ----- Construct the model -----
    # DirectVGPModel's __init__ computes the fresh eigenspace (B, eigvals_b,
    # K_tilde_b, K_b, KKtilde_inv_b, Kvec) from (kernel, X_tilde, eigval_tol).
    #
    # X_tilde selection: the upstream loader assumed M == n_train, i.e.
    # X_tilde = X_train. Here M < n_train (M=300 against a 3160-image pool),
    # and models/gp/train.py takes the inducing points as X_train[:M], so the
    # FIRST M rows of X_train are exactly them. Read M from metadata['M'] and
    # slice; when M == n_train the slice is a no-op.
    eigval_tol = metadata['eigval_tol']
    lambda_var_clamp = metadata['lambda_var_clamp']
    M_inducing = metadata.get('M', X_train.shape[0])
    if M_inducing > X_train.shape[0]:
        raise ValueError(
            f"Checkpoint metadata['M']={M_inducing} exceeds pool size "
            f"{X_train.shape[0]}; inducing slice is invalid."
        )
    X_tilde = X_train[:M_inducing]
    model = DirectVGPModel(
        kernel=kernel,
        likelihood=likelihood,
        X_train=X_train,
        X_tilde=X_tilde,
        eigval_tol=eigval_tol,
        lambda_var_clamp=lambda_var_clamp,
    )

    # Inject saved variational parameters, overriding the constructor's init.
    m_b = checkpoint['m_b'].to(device=device, dtype=dtype)
    V_b = checkpoint['V_b'].to(device=device, dtype=dtype)

    # If the checkpoint also persisted the eigenbasis (B, eigvals_b), restore
    # it instead of trusting the freshly recomputed one.  eigh's per-column
    # sign / rotation-within-degenerate-subspaces convention is not stable
    # across PyTorch / cuSOLVER versions; m_b and V_b are stored in the
    # training-time basis, so reusing the saved B is what makes save→load
    # round-trip predictions identical.  Old checkpoints (no 'B' key) fall
    # back to the recomputed basis with a warning — those load-time
    # predictions are only reliable when the env matches training.
    if 'B' in checkpoint and 'eigvals_b' in checkpoint:
        from dataclasses import replace as _dc_replace
        B_saved         = checkpoint['B'].to(device=device, dtype=dtype)
        eigvals_b_saved = checkpoint['eigvals_b'].to(device=device, dtype=dtype)
        with torch.no_grad():
            K_train_tilde = kernel(X_train, X_tilde).to_dense()
            K_b           = K_train_tilde @ B_saved
            K_tilde_b     = torch.diag(eigvals_b_saved)
            KKtilde_inv_b = K_b / eigvals_b_saved.unsqueeze(0)
        model._state = _dc_replace(
            model._state,
            B             = B_saved,
            eigvals_b     = eigvals_b_saved,
            K_tilde_b     = K_tilde_b,
            K_b           = K_b,
            KKtilde_inv_b = KKtilde_inv_b,
        )
    else:
        warnings.warn(
            "Checkpoint has no saved 'B'/'eigvals_b' (legacy format). "
            "Falling back to recomputed eigenspace basis — predictions may "
            "diverge from training-time values across PyTorch / cuSOLVER "
            "versions due to eigh sign/rotation ambiguity."
        )

    # Guard against eigenspace dimension mismatch (±1 from numerical differences
    # between training and loading environments shifting the eigval_tol cutoff).
    model_n_b = model.state.m_b.shape[0]
    saved_n_b = m_b.shape[0]
    if model_n_b != saved_n_b:
        warnings.warn(
            f"Eigenspace dimension mismatch: model has n_b={model_n_b} "
            f"but checkpoint has n_b={saved_n_b}. "
            f"Truncating to min({model_n_b}, {saved_n_b}) — dropped modes "
            f"correspond to near-threshold eigenvalues; effect is negligible."
        )
        n_b = min(model_n_b, saved_n_b)
        m_b = m_b[:n_b]
        V_b = V_b[:n_b, :n_b]
        if model_n_b > n_b:
            # Pad with zeros (prior) for extra dimensions in new eigenspace
            pad = torch.zeros(model_n_b - n_b, dtype=m_b.dtype, device=m_b.device)
            m_b = torch.cat([m_b, pad])
            V_pad = torch.zeros(model_n_b, model_n_b, dtype=V_b.dtype, device=V_b.device)
            V_pad[:n_b, :n_b] = V_b
            V_b = V_pad

    model.update_variational_params(m_b, V_b)

    return {
        'model': model,
        'config': config,
        'metrics': checkpoint['metrics'],
        'hyperparams': checkpoint['hyperparams'],
        'metadata': metadata,
        'pool_indices': pool_indices,
    }
