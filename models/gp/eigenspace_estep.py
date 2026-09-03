"""
E-Step for Eigenspace Variational GP

Implements Newton update for variational parameters (m_b, V_b) in the eigenspace
representation used by DirectVGPModel.

Key functions:
- estep_eigenspace: Newton update in eigenspace

Extracted from estep.py during codebase reorganization (2025-02).
"""

import torch


# =============================================================================
# Eigenspace E-Step
# =============================================================================

def estep_eigenspace(model, r: torch.Tensor, f_mean: torch.Tensor) -> None:
    """Newton update for variational parameters in eigenspace.

    This implementation matches varGP utils.py:Estep() lines 4244-4256 exactly.
    Runs entirely under torch.no_grad() — the Newton update is closed-form
    and never needs autograd. Without no_grad, the computation graph chains
    through A (which has requires_grad) and causes unbounded memory growth.

    The g_b and G_b computed here are the TRANSFORMED versions:
        g_b = K_tilde_inv @ g_standard
        G_b = K_tilde_inv @ G_standard @ K_tilde_inv

    The formulas below are correct for these transformed quantities.

    Note on m_new: this reproduces the update of the reference implementation,
        m_new = V_new @ (G @ m + g)
    which differs from
        m_new = m + K_tilde @ solve(K_tilde + G, g - m)
    unless K_tilde and G commute. The reference form is kept deliberately, so
    that the baseline matches the published one.

    Args:
        model: DirectVGPModel instance
        r: Spike counts, shape (N,)
        f_mean: Expected firing rate exp(A*lambda_m + 0.5*A^2*lambda_var + lambda0),
                shape (N,)
    """
    with torch.no_grad():
        state = model.state
        A = model.likelihood.A.squeeze()

        a = state.KKtilde_inv_b  # (N, n_b) - this is K @ K_tilde_inv in eigenspace

        # Transformed gradient: g_b = A * a.T @ (r - f_mean)
        # This is K_tilde_inv @ g_standard
        g_b = A * (a.T @ (r - f_mean))  # (n_b,)

        # Transformed Hessian: G_b = A^2 * a.T @ diag(f_mean) @ a
        # This is K_tilde_inv @ G_standard @ K_tilde_inv
        G_b = (A * A) * (a.T @ (f_mean[:, None] * a))  # (n_b, n_b)

        # V update: V_new = solve(I + K_tilde @ G, K_tilde)
        # Matches varGP utils.py line 4246
        n_b = state.K_tilde_b.shape[0]
        eye = torch.eye(n_b, dtype=state.K_tilde_b.dtype, device=state.K_tilde_b.device)
        V_b_new = torch.linalg.solve(eye + state.K_tilde_b @ G_b, state.K_tilde_b)

        # m update: m_new = V_new @ (G @ m + g)
        # Matches varGP utils.py line 4247
        m_b_new = V_b_new @ (G_b @ state.m_b + g_b)

        # Symmetrize V for numerical stability
        V_b_new = (V_b_new + V_b_new.T) / 2

        model.update_variational_params(m_b_new, V_b_new)
