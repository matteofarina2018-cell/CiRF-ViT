"""
Eigenspace Training Functions for DirectVGPModel

This module contains training, evaluation, and prediction functions for the
eigenspace-based variational GP implementation (DirectVGPModel).

Key functions:
- train_eigenspace: Main EM-style training loop
- compute_elbo_eigenspace: ELBO computation in eigenspace
- predict_eigenspace: Prediction at test points

Extracted from train.py during codebase reorganization (2025-02).
"""

import time
import warnings
from typing import Dict

import torch

from .metrics import compute_pearson_correlation, compute_spearman_correlation


def compute_elbo_eigenspace(
    state,  # DirectVariationalState
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    A: torch.Tensor,
    lambda0: torch.Tensor
) -> torch.Tensor:
    """Compute ELBO = log_likelihood - KL_divergence.

    Log-likelihood:
        L = sum(r * (A * lambda_m + lambda0) - f_mean)
        where f_mean = exp(A * lambda_m + 0.5 * A^2 * lambda_var + lambda0)

    KL divergence for q(u) = N(m, V) vs p(u) = N(0, K_tilde):
        KL = 0.5 * (tr(K_tilde^-1 @ V) + m.T @ K_tilde^-1 @ m - n_b + log|K_tilde| - log|V|)

    In eigenspace with K_tilde_b diagonal:
        KL = 0.5 * (sum(V_b_diag / eigvals) + sum(m_b^2 / eigvals) - n_b
                   + sum(log(eigvals)) - log|V_b|)

    Args:
        state: Current DirectVariationalState
        r: Spike counts, shape (N,)
        lambda_m: Posterior mean, shape (N,)
        lambda_var: Posterior variance, shape (N,)
        A: Gain parameter (scalar)
        lambda0: Bias parameter (scalar)

    Returns:
        ELBO value (scalar, to be maximized)
    """
    # Import here to avoid circular dependency
    from .utils import compute_f_mean

    # Log-likelihood
    f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)
    log_lik = (r * (A * lambda_m + lambda0) - f_mean).sum()

    # KL divergence in eigenspace
    n_b = len(state.eigvals_b)
    eigvals = state.eigvals_b

    # tr(K_tilde^-1 @ V) = tr(diag(1/eigvals) @ V_b) = sum(V_b_diag / eigvals)
    V_diag = torch.diag(state.V_b)
    trace_term = (V_diag / eigvals).sum()

    # m.T @ K_tilde^-1 @ m = sum(m_b^2 / eigvals)
    quad_term = ((state.m_b ** 2) / eigvals).sum()

    # log|K_tilde| = sum(log(eigvals))
    log_det_K = torch.log(eigvals).sum()

    # log|V| via eigh (slogdet unsupported on Blackwell GPUs in current torch builds)
    # V_b is symmetric PD by construction, so all eigenvalues are positive.
    eigvals_V = torch.linalg.eigh(state.V_b).eigenvalues
    if (eigvals_V <= 0).any():
        warnings.warn("V_b is not positive definite in KL computation")
        log_det_V = torch.tensor(0.0, device=state.V_b.device, dtype=state.V_b.dtype)
    else:
        log_det_V = torch.log(eigvals_V).sum()

    KL = 0.5 * (trace_term + quad_term - n_b + log_det_K - log_det_V)

    return log_lik - KL


def _compute_val_metrics(model, X_val, r_val):
    """Compute validation log-likelihood, Pearson r, and Spearman rho in one pass.

    Val log-lik formula (same as ELBO's log-lik term on held-out data):
      val_ll = sum(r_val * (A*mu + lambda0) - f_mean)
    where f_mean = exp(A*mu + 0.5*A^2*var + lambda0)

    Val Pearson r: corr(f_pred, r_val) — same metric as test_r.

    Args:
        model: DirectVGPModel instance (trained or mid-training)
        X_val: Validation images, shape (N_val, n_features)
        r_val: Validation spike counts for one cell, shape (N_val,)

    Returns:
        Tuple (val_ll, val_r, val_rho):
          val_ll: Scalar float, validation expected log-likelihood (higher is better)
          val_r: Scalar float, Pearson correlation between f_pred and r_val
          val_rho: Scalar float, Spearman rank correlation between f_pred and r_val
    """
    with torch.no_grad():
        preds = predict_eigenspace(model, X_val)
        A = model.likelihood.A.squeeze()
        lambda0 = model.likelihood.lambda0.squeeze()
        f_pred = preds['f_pred']
        val_ll = (r_val * (A * preds['lambda_m'] + lambda0) - f_pred).sum().item()
        val_r = compute_pearson_correlation(r_val, f_pred)
        val_rho = compute_spearman_correlation(r_val, f_pred)
    return val_ll, val_r, val_rho


def _save_model_state(model):
    """Save model state for restore-best early stopping.

    Saves variational parameters (m_b, V_b), the eigenspace basis B,
    and all kernel/likelihood raw parameters. B is needed because
    recompute_eigenspace() reprojects (m_b, V_b) from old B to new B,
    and the saved m_b/V_b dimensions must match the saved B.
    """
    return {
        'm_b': model.state.m_b.clone(),
        'V_b': model.state.V_b.clone(),
        'B': model.state.B.clone(),
        'kernel_state': {n: p.detach().clone()
                         for n, p in model.kernel.named_parameters()},
        'likelihood_state': {n: p.detach().clone()
                             for n, p in model.likelihood.named_parameters()},
    }


def _restore_model_state(model, saved):
    """Restore model state from a previous save.

    Restores kernel/likelihood parameters first, then sets the eigenspace
    basis B and variational params (m_b, V_b) from the save, and finally
    recomputes the eigenspace. The saved B ensures the reprojection inside
    recompute_eigenspace has matching dimensions.
    """
    with torch.no_grad():
        # Restore kernel/likelihood params first
        for n, p in model.kernel.named_parameters():
            p.copy_(saved['kernel_state'][n])
        for n, p in model.likelihood.named_parameters():
            p.copy_(saved['likelihood_state'][n])
        # Restore eigenspace basis and variational params (matching dims)
        model.state.B = saved['B'].clone()
        model.update_variational_params(saved['m_b'], saved['V_b'])
    # Recompute eigenspace from restored kernel params
    # (reprojects saved m_b, V_b via saved B into new eigenspace)
    model.recompute_eigenspace()


def train_eigenspace(
    model,  # DirectVGPModel
    r: torch.Tensor,
    n_iterations: int,
    n_estep: int,
    n_fstep: int,
    n_mstep: int,
    lr_f: float,
    lr_m: float,
    print_every: int = 10,
    verbose: bool = False,
    use_analytical_mstep: bool = False,
    capture_checkpoints: bool = False,
    early_stop: bool = True,
    patience: int = 15,
    min_delta_rel: float = 0.001,
    min_iterations: int = 10,
    restore_best: bool = True,
    f_mean_max_threshold: float = 500,
    f_mean_mean_threshold: float = 100,
    fix_Amp: bool = False,
    interleave_fstep: bool = False,
    X_val: torch.Tensor = None,
    r_val: torch.Tensor = None,
    es_metric: str = 'elbo',
    collect_mstep_diagnostics: bool = False,
) -> Dict:
    """Train using eigenspace-based variational GP - model-based API.

    Uses DirectVGPModel which owns kernel, likelihood, and state.

    Args:
        model: DirectVGPModel instance (owns kernel, likelihood, X, X_tilde, state)
        r: Spike counts, shape (N,)
        n_iterations: Maximum number of EM iterations
        n_estep: Number of E-step Newton iterations per EM iteration
        n_fstep: Number of F-step LBFGS iterations
        n_mstep: Number of M-step LBFGS iterations
        lr_f: Learning rate for F-step
        lr_m: Learning rate for M-step
        print_every: Print progress every N iterations
        verbose: Print detailed debugging info
        use_analytical_mstep: Use analytical gradients for M-step
        capture_checkpoints: If True, capture state at each checkpoint for validation
        early_stop: Enable early stopping based on validation log-lik (default: True)
        patience: Iterations without sufficient validation improvement before stopping
        min_delta_rel: Minimum relative improvement to reset patience counter
        min_iterations: Minimum iterations before early stopping can trigger (default: 10)
        restore_best: Restore model to best-validation-iteration on early stop
        f_mean_max_threshold: Max f_mean.max() before step rejection (default: 500)
        f_mean_mean_threshold: Max f_mean.mean() before step rejection (default: 100)
        X_val: Validation images, shape (N_val, n_features). None = no validation.
        r_val: Validation spike counts, shape (N_val,). None = no validation.
        collect_mstep_diagnostics: If True, collect per-M-step LBFGS telemetry
            (closure calls, ELBO trace, timing, termination reason) for the
            autograd M-step path. Off by default. Only supported for the
            autograd M-step path — warns and collects nothing if
            use_analytical_mstep=True.

    Returns:
        Dict with:
            'losses': List of ELBO values per iteration
            'model': Trained DirectVGPModel
            'time_estep_total': Total E-step time (includes F-step)
            'time_mstep_total': Total M-step time
            'stopped_early': Whether training stopped early
            'final_iteration': Final iteration number
            'best_iteration': Iteration with the highest ES metric value (ELBO by default)
            'curves': Dict of per-iteration curves (train_loss, val_log_lik, params, etc.)
            'checkpoints': List of checkpoint dicts (only if capture_checkpoints=True)
            'mstep_diagnostics': List of per-outer-iteration M-step diagnostic
                dicts, or None if collect_mstep_diagnostics=False. Each entry:
                  {'outer_iter': int, 'n_lbfgs_iters': int, 'termination': str,
                   'total_time_s': float, 'closure_calls': [...]}
    """
    from .eigenspace_estep import estep_eigenspace
    from .eigenspace_fstep import fstep_eigenspace, damped_newton_update_A_lambda0
    from .eigenspace_mstep import mstep_eigenspace_autograd, mstep_eigenspace_analytical
    from .utils import compute_f_mean

    if capture_checkpoints:
        # Development-only hook: capture_checkpoint() lives in the authors'
        # restructuring-invariance test harness, which is not part of this
        # package. Nothing in the published pipeline sets this flag.
        raise NotImplementedError(
            'capture_checkpoints=True needs the development test harness, which '
            'is not distributed with this repository; train with the default '
            'capture_checkpoints=False.')

    state = model.state
    n_b = len(state.eigvals_b)
    M = model.X_tilde.shape[0]
    print(f"Eigenspace dimension: n_b={n_b} (from M={M} inducing points)")

    # Freeze Amp if requested (paper's code has no Amp parameter)
    if fix_Amp:
        model.kernel.raw_Amp.requires_grad_(False)
        print(f"  Amp FROZEN at {model.kernel.Amp.item():.4f} (fix_Amp=True)")

    if interleave_fstep:
        print(f"  F-step INTERLEAVED: damped Newton (alpha=0.25) at each E-step iteration")

    time_estep_total = 0.0
    time_mstep_total = 0.0
    losses = []
    checkpoints = [] if capture_checkpoints else None

    if collect_mstep_diagnostics and use_analytical_mstep:
        import warnings as _warnings
        _warnings.warn(
            "collect_mstep_diagnostics=True with use_analytical_mstep=True: "
            "analytical M-step path is not instrumented. No diagnostics will "
            "be recorded."
        )
    mstep_diagnostics_list = [] if collect_mstep_diagnostics else None

    # Early stopping state
    stopped_early = False
    final_iteration = 0
    has_val = X_val is not None and r_val is not None
    # `best_es_value`: the true argmax of the metric — updates on ANY improvement,
    # used for restore_best. Matches PyTorch Lightning's ModelCheckpoint semantics.
    best_es_value = float('-inf')
    # `patience_reference`: the value at the last "meaningful" (> min_delta_rel)
    # improvement — used only for patience counter resets. Decoupled from
    # best tracking so slow steady growth can accumulate and still trigger
    # a reset when cumulative improvement crosses the threshold.
    patience_reference = float('-inf')
    patience_counter = 0
    best_state = None
    best_iteration = 0

    # Curve storage (logged every iteration regardless of early_stop setting)
    train_loss_curve = []
    train_ll_curve = []
    train_kl_curve = []
    val_ll_curve = []
    train_r_curve = []
    val_r_curve = []
    train_rho_curve = []
    val_rho_curve = []
    param_A_curve = []
    param_lambda0_curve = []
    param_beta_curve = []
    param_rho_curve = []
    param_sigma_0_curve = []
    param_eps_0x_curve = []
    param_eps_0y_curve = []
    param_Amp_curve = []
    iter_time_curve = []

    # Initial moments (GPyTorch-like: call model to get posterior).
    # no_grad: posterior is for E-step input, not for backprop.
    # A/lambda0 detached: they are constants in the E-step/metrics context.
    # The F-step and M-step read from model.likelihood directly.
    with torch.no_grad():
        posterior = model(model.X_train)
        lambda_m, lambda_var = posterior.mean, posterior.variance
    A = model.likelihood.A.squeeze().detach()
    lambda0 = model.likelihood.lambda0.squeeze().detach()
    f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

    for iteration in range(1, n_iterations):
        iter_start_time = time.time()

        # ===== Kernel recomputation after M-step =====
        if n_mstep > 0 and iteration > 1:
            model.recompute_eigenspace()
            with torch.no_grad():
                posterior = model(model.X_train)
                lambda_m, lambda_var = posterior.mean, posterior.variance
            A = model.likelihood.A.squeeze().detach()
            lambda0 = model.likelihood.lambda0.squeeze().detach()
            f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

            if capture_checkpoints:
                checkpoints.append(capture_checkpoint(
                    'C1_eigenspace', iteration, model.state,
                    lambda_m=lambda_m, lambda_var=lambda_var, f_mean=f_mean
                ))

        # ===== E-step: Newton loop =====
        start_estep = time.time()

        for i_estep in range(n_estep):
            # Save state before Newton step for revert on divergence
            m_b_prev = model.state.m_b.clone()
            V_b_prev = model.state.V_b.clone()
            if interleave_fstep:
                raw_A_prev = model.likelihood.raw_A.detach().clone()
                lambda0_prev = model.likelihood.lambda0.detach().clone()

            estep_eigenspace(model, r, f_mean)

            with torch.no_grad():
                posterior = model(model.X_train)
                lambda_m, lambda_var = posterior.mean, posterior.variance

            if interleave_fstep:
                # Paper's approach: update A, lambda0 at every E-step iteration
                f_mean = damped_newton_update_A_lambda0(
                    model, r, lambda_m, lambda_var,
                    f_mean_max_threshold=f_mean_max_threshold,
                    f_mean_mean_threshold=f_mean_mean_threshold,
                )
                A = model.likelihood.A.squeeze().detach()
                lambda0 = model.likelihood.lambda0.squeeze().detach()
            else:
                f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

            if capture_checkpoints:
                checkpoints.append(capture_checkpoint(
                    'C2_estep', iteration, model.state,
                    lambda_m=lambda_m, lambda_var=lambda_var, f_mean=f_mean,
                    estep_idx=i_estep
                ))

            # Check for divergence: mean threshold catches global blowup,
            # max threshold catches localized blowup. Both trigger revert.
            if (f_mean.mean().item() > f_mean_mean_threshold
                    or f_mean.max().item() > f_mean_max_threshold
                    or torch.any(torch.isnan(f_mean))):
                model.update_variational_params(m_b_prev, V_b_prev)
                if interleave_fstep:
                    with torch.no_grad():
                        model.likelihood.raw_A.copy_(raw_A_prev)
                        model.likelihood.lambda0.copy_(lambda0_prev)
                    A = model.likelihood.A.squeeze().detach()
                    lambda0 = model.likelihood.lambda0.squeeze().detach()
                with torch.no_grad():
                    posterior = model(model.X_train)
                    lambda_m, lambda_var = posterior.mean, posterior.variance
                f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)
                print(f"  E-step {i_estep}: f_mean diverged "
                      f"(max={f_mean.max().item():.1f}), reverted")
                break

        # ===== F-step: Optimize A =====
        if not interleave_fstep:
            fstep_eigenspace(model, r, lambda_m, lambda_var, n_fstep, lr_f,
                             f_mean_max_threshold=f_mean_max_threshold,
                             f_mean_mean_threshold=f_mean_mean_threshold)

        A = model.likelihood.A.squeeze().detach()
        lambda0 = model.likelihood.lambda0.squeeze().detach()
        f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

        if capture_checkpoints:
            checkpoints.append(capture_checkpoint(
                'C3_fstep', iteration, model.state,
                likelihood=model.likelihood,
                lambda_m=lambda_m, lambda_var=lambda_var, f_mean=f_mean
            ))

        time_estep_total += time.time() - start_estep

        # ===== Compute and record loss =====
        # NOTE: Metrics are computed BEFORE M-step so that all quantities
        # (lambda_m, lambda_var, f_mean, kernel params, eigenspace) are from
        # the same consistent model state. The M-step changes kernel params
        # without recomputing the eigenspace, so predict_eigenspace() would
        # give inconsistent results if called after M-step.
        elbo = compute_elbo_eigenspace(model.state, r, lambda_m, lambda_var, A, lambda0)

        # Detect training divergence (NaN/inf loss)
        if torch.isnan(elbo) or torch.isinf(elbo):
            print(f"Training diverged at iteration {iteration}: elbo={elbo.item()}")
            stopped_early = True
            break

        loss = -elbo.item()
        losses.append(loss)
        final_iteration = iteration

        # Training log-lik (same formula as ELBO's lik term, for curve logging)
        train_ll = (r * (A * lambda_m + lambda0) - f_mean).sum().item()
        # KL from ELBO decomposition: ELBO = log_lik - KL => KL = log_lik - ELBO
        train_kl = train_ll - elbo.item()
        # Training correlations: same metrics as test_r
        train_r = compute_pearson_correlation(r, f_mean.detach())
        train_rho = compute_spearman_correlation(r, f_mean.detach())

        # ===== Validation evaluation =====
        val_ll = None
        val_r = None
        val_rho = None
        if has_val:
            val_ll, val_r, val_rho = _compute_val_metrics(model, X_val, r_val)

        # ===== Log curves =====
        # iter_elapsed does not include M-step time (M-step runs after logging);
        # M-step time is tracked separately in time_mstep_total.
        iter_elapsed = time.time() - iter_start_time
        train_loss_curve.append(loss)
        train_ll_curve.append(train_ll)
        train_kl_curve.append(train_kl)
        val_ll_curve.append(val_ll)
        train_r_curve.append(train_r)
        val_r_curve.append(val_r)
        train_rho_curve.append(train_rho)
        val_rho_curve.append(val_rho)
        param_A_curve.append(A.item())
        param_lambda0_curve.append(lambda0.item())
        param_beta_curve.append(model.kernel.beta.item())
        param_rho_curve.append(model.kernel.rho.item())
        param_sigma_0_curve.append(model.kernel.sigma_0.item())
        param_eps_0x_curve.append(model.kernel.eps_0x.item())
        param_eps_0y_curve.append(model.kernel.eps_0y.item())
        param_Amp_curve.append(model.kernel.Amp.item())
        iter_time_curve.append(iter_elapsed)

        if capture_checkpoints:
            checkpoints.append(capture_checkpoint(
                'C5_end', iteration, model.state,
                loss=loss
            ))

        if iteration % print_every == 0 or iteration == 1:
            val_str = f", val_ll={val_ll:.2f}" if val_ll is not None else ""
            val_r_str = f", val_r={val_r:.4f}" if val_r is not None else ""
            val_rho_str = f", val_rho={val_rho:.4f}" if val_rho is not None else ""
            print(f"Iter {iteration}/{n_iterations-1}: loss={loss:.2f}{val_str}{val_r_str}{val_rho_str}, "
                  f"A={A.item():.4f}, lambda0={lambda0.item():.4f}, "
                  f"n_b={len(model.state.eigvals_b)}")

        # ===== Patience-based early stopping on ELBO =====
        # ELBO is the only supported ES metric: under the interleaved F-step
        # the transient in A makes val_ll / val_r / val_rho too noisy to stop on.
        # Use es_metric='none' to disable early stopping entirely.
        es_value = None
        if es_metric == 'elbo':
            es_value = -loss  # ELBO = -train_loss (higher is better)
        elif es_metric == 'none':
            es_value = None  # ES disabled
        else:
            raise ValueError(
                f"Unknown es_metric: {es_metric!r}. "
                f"Valid choices are 'elbo' (default) or 'none' (disabled). "
                f"val_ll/val_r/val_rho are not supported: the A transient "
                f"under the interleaved F-step makes them too noisy to stop on."
            )

        if es_value is not None:
            # Two separate concerns tracked independently (matches PyTorch
            # Lightning's design, where ModelCheckpoint and EarlyStopping are
            # decoupled, rather than Keras's conflated single callback):
            #
            # 1) BEST TRACKING: update best_es_value on ANY improvement.
            #    Used by restore_best to revert to the true argmax of the
            #    metric. For (near-)monotonic metrics like ELBO, this
            #    prevents losing slow-steady progress to the patience threshold.
            #
            # 2) PATIENCE COUNTING: reset patience_counter only on
            #    "meaningful" improvements (cumulative rel gain > min_delta_rel
            #    since the last reset). Compared against patience_reference,
            #    NOT best_es_value — so slow steady growth accumulates against
            #    a fixed reference and eventually resets.
            is_first = (best_es_value == float('-inf'))

            # (1) Best tracking — always update on any improvement.
            if is_first or es_value > best_es_value:
                best_es_value = es_value
                best_iteration = iteration
                if restore_best:
                    best_state = _save_model_state(model)

            # (2) Patience counter — reset only on meaningful cumulative gain.
            if is_first:
                patience_reference = es_value
                patience_counter = 0
            else:
                rel_improvement = (es_value - patience_reference) / max(abs(patience_reference), 1e-8)
                if rel_improvement > min_delta_rel:
                    patience_reference = es_value
                    patience_counter = 0
                else:
                    patience_counter += 1

            if early_stop and patience_counter >= patience and iteration >= min_iterations:
                if restore_best and best_state is not None:
                    _restore_model_state(model, best_state)
                    posterior = model(model.X_train)
                    lambda_m, lambda_var = posterior.mean, posterior.variance
                stopped_early = True
                print(f"Early stopping at iteration {iteration} (metric={es_metric}): "
                      f"no improvement for {patience} iters "
                      f"(best={best_es_value:.4f} at iter {best_iteration})"
                      f"{', restored best' if restore_best else ''}")
                break

        # ===== M-step: Optimize kernel hyperparameters =====
        # Runs AFTER metrics/val_ll/early_stopping to ensure all evaluations
        # use a consistent model state. The eigenspace is recomputed at the
        # START of the next iteration to sync with the new kernel params.
        start_mstep = time.time()

        if n_mstep > 0 and iteration < n_iterations - 1:
            if use_analytical_mstep:
                mstep_eigenspace_analytical(model, r, n_mstep, lr_m,
                                            f_mean_mean_threshold=f_mean_mean_threshold,
                                            lambda_var_clamp=model.lambda_var_clamp)
            else:
                _mstep_diag = mstep_eigenspace_autograd(
                    model, r, n_mstep, lr_m,
                    f_mean_mean_threshold=f_mean_mean_threshold,
                    lambda_var_clamp=model.lambda_var_clamp,
                    collect_diagnostics=collect_mstep_diagnostics,
                )
                if collect_mstep_diagnostics and _mstep_diag is not None:
                    _mstep_diag['outer_iter'] = iteration
                    mstep_diagnostics_list.append(_mstep_diag)

            if capture_checkpoints:
                checkpoints.append(capture_checkpoint(
                    'C4_mstep', iteration, model.state,
                    kernel=model.kernel
                ))

        time_mstep_total += time.time() - start_mstep

    # Fallback: if ES was disabled (es_metric='none') or never tracked a
    # best iteration, report the final iteration as "best" so downstream
    # code has a sensible value. Covers the es_metric='none' + n_val_split=0
    # combo where best_iteration would otherwise stay at 0.
    if not stopped_early and best_iteration == 0:
        best_iteration = final_iteration

    result = {
        'losses': losses,
        'model': model,
        'time_estep_total': time_estep_total,
        'time_mstep_total': time_mstep_total,
        'stopped_early': stopped_early,
        'final_iteration': final_iteration,
        'best_iteration': best_iteration,
        'curves': {
            'train_loss': train_loss_curve,
            'train_log_lik': train_ll_curve,
            'train_kl': train_kl_curve,
            'val_log_lik': val_ll_curve,
            'train_r': train_r_curve,
            'val_r': val_r_curve,
            'train_rho': train_rho_curve,
            'val_rho': val_rho_curve,
            'A': param_A_curve,
            'lambda0': param_lambda0_curve,
            'beta': param_beta_curve,
            'rho': param_rho_curve,
            'sigma_0': param_sigma_0_curve,
            'eps_0x': param_eps_0x_curve,
            'eps_0y': param_eps_0y_curve,
            'Amp': param_Amp_curve,
            'iter_time': iter_time_curve,
        },
    }
    if capture_checkpoints:
        result['checkpoints'] = checkpoints
    result['mstep_diagnostics'] = mstep_diagnostics_list
    return result


def predict_eigenspace(model, X_test: torch.Tensor) -> Dict:
    """Predict at test points using trained eigenspace variational GP.

    Computes posterior moments at test points and expected firing rates.

    Args:
        model: Trained DirectVGPModel
        X_test: Test inputs, shape (N_test, n_features)

    Returns:
        Dict with:
            'f_pred': Predicted firing rates, shape (N_test,)
            'lambda_m': Posterior mean at test points, shape (N_test,)
            'lambda_var': Posterior variance at test points, shape (N_test,)
    """
    kernel = model.kernel
    likelihood = model.likelihood
    state = model.state
    X_tilde = model.X_tilde

    with torch.no_grad():
        # Compute cross-kernel to inducing points
        K_test = kernel(X_test, X_tilde).to_dense()  # (N_test, M)
        Kvec_test = kernel(X_test, diag=True)  # (N_test,)

        # Project to eigenspace
        K_test_b = K_test @ state.B  # (N_test, n_b)

        # a = K_test @ K_tilde_inv = K_test_b @ diag(1/eigvals)
        a = K_test_b / state.eigvals_b.unsqueeze(0)  # (N_test, n_b)

        # Posterior mean: lambda_m = a @ m_b
        lambda_m = a @ state.m_b  # (N_test,)

        # Posterior variance
        V_minus_K = state.V_b - state.K_tilde_b
        aV = a @ V_minus_K
        lambda_var = Kvec_test + (a * aV).sum(dim=1)
        lambda_var = torch.clamp(lambda_var, min=model.lambda_var_clamp)

        # Predicted firing rate
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_pred = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

    return {
        'f_pred': f_pred,
        'lambda_m': lambda_m,
        'lambda_var': lambda_var,
    }
