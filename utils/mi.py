"""Gaussian mutual information estimation via log-determinant closed form.

For two multivariate representations X (B, D_x) and Y (B, D_y), assuming
joint Gaussianity:

    I(X; Y) = ½ ( log|Σ_xx| + log|Σ_yy| − log|Σ| )

where Σ is the (D_x + D_y, D_x + D_y) joint covariance matrix and Σ_xx, Σ_yy
are the marginal blocks.  The estimate is the plug-in MLE: we substitute the
sample covariance for the population covariance.

Using Cholesky decomposition for numerical stability and adding a small
diagonal jitter to guard against singular covariance matrices when the
batch contains near-constant directions.
"""

import torch


def _cov(X):
    """Sample covariance matrix — unbiased Bessel correction (N − 1).

    Args:
        X: (B, D)

    Returns:
        cov: (D, D)
    """
    B = X.shape[0]
    if B < 2:
        raise ValueError(f"Gaussian MI requires batch size ≥ 2, got {B}")
    Xc = X - X.mean(dim=0, keepdim=True)
    return (Xc.t() @ Xc) / (B - 1)


def _logdet(X):
    """log-determinant via Cholesky, with a small jitter for numerical safety.

    Args:
        X: (D, D) symmetric positive semi-definite matrix

    Returns:
        logdet: scalar
    """
    jitter = 1e-6 * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
    L = torch.linalg.cholesky(X + jitter)
    return 2.0 * L.diagonal().log().sum()


def gaussian_mi(x, y, jitter=1e-6):
    """Gaussian mutual information I(x; y) in nats.

    Args:
        x: (B, D_x) tensor
        y: (B, D_y) tensor
        jitter: diagonal regularisation added to each covariance block
                to prevent numerical singularities

    Returns:
        mi: non-negative scalar (clamped to ≥ 0 to mask tiny negative
            values that can arise from finite-sample covariance noise)
    """
    B = x.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # ── joint covariance ──
    Z = torch.cat([x, y], dim=-1)                     # (B, D_x + D_y)
    S = _cov(Z)                                       # (D_total, D_total)
    S_xx = S[: x.shape[-1], : x.shape[-1]]
    S_yy = S[x.shape[-1] :, x.shape[-1] :]

    eps = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device) * jitter
    mi = 0.5 * (_logdet(S_xx + eps) +
                _logdet(S_yy + eps) -
                _logdet(S + jitter * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)))

    # Clamp to [0, ∞) — finite-sample covariance can produce tiny negative estimates
    return mi.clamp(min=0.0)


def gaussian_mi_per_client(x, y):
    """Per-client Gaussian MI for batched tensors (M, B, D).

    When M is small (typical 12–50), a Python loop is negligible overhead
    compared to the vmap forward pass.  Falls back to calling gaussian_mi
    for each client independently.

    Args:
        x: (M, B, D) stacked representations
        y: (M, B, D)

    Returns:
        mi: (M,) tensor, one MI value per client
    """
    M = x.shape[0]
    out = torch.empty(M, dtype=x.dtype, device=x.device)
    for m in range(M):
        out[m] = gaussian_mi(x[m], y[m])
    return out
