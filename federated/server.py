"""Federated server: prototype aggregation via FedAvg, with optional OT alignment."""

import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is expected, but degrade gracefully
    _HAS_SCIPY = False


def _sinkhorn(cost, eps, iters):
    """Balanced entropic-OT coupling between two K-point uniform marginals.

    Args:
        cost: (K, K) transport cost (here 1 - cosine_sim).
        eps: entropic regularization strength.
        iters: number of Sinkhorn iterations.

    Returns:
        T: (K, K) coupling matrix; both marginals are uniform 1/K, so every row
           and column sums to 1/K.
    """
    K = cost.shape[0]
    # Subtract the min for numerical stability; an additive shift on the cost
    # leaves the optimal coupling unchanged.
    Kmat = torch.exp(-(cost - cost.min()) / eps)  # Gibbs kernel
    a = torch.full((K,), 1.0 / K, dtype=cost.dtype, device=cost.device)
    u = torch.ones(K, dtype=cost.dtype, device=cost.device)
    v = torch.ones(K, dtype=cost.dtype, device=cost.device)
    for _ in range(iters):
        u = a / (Kmat @ v + 1e-12)
        v = a / (Kmat.t() @ u + 1e-12)
    return u.unsqueeze(1) * Kmat * v.unsqueeze(0)  # diag(u) Kmat diag(v)


def _align_one(anchor, proto, mode, eps, iters):
    """Align one client's prototypes to the global anchor by index.

    Args:
        anchor: (K, D) reference prototypes (this round's broadcast global).
        proto: (K, D) one client's locally-trained prototypes.
        mode: "ot" (soft Sinkhorn barycenter) or "hungarian" (hard permutation).
        eps, iters: Sinkhorn hyperparameters (ignored for hungarian).

    Returns:
        aligned: (K, D) client prototypes reordered/recombined so that row k
                 corresponds to anchor row k.
        perm_like: scalar in [0, 1], how far the matching is from the identity
                   (0 = already aligned, larger = more reordering).
    """
    g = F.normalize(anchor, dim=-1, eps=1e-6)
    p = F.normalize(proto, dim=-1, eps=1e-6)
    sim = g @ p.t()          # (K, K) cosine sim: anchor row i vs client row j
    cost = 1.0 - sim         # in [0, 2]
    K = cost.shape[0]

    if mode == "hungarian":
        if not _HAS_SCIPY:
            raise RuntimeError("proto_align='hungarian' requires scipy")
        # row_ind is 0..K-1 sorted; col_ind[i] is the client slot matched to
        # anchor slot i. Reordering proto by col_ind aligns it to the anchor.
        _, col = linear_sum_assignment(cost.detach().cpu().numpy())
        col = torch.as_tensor(col, dtype=torch.long, device=proto.device)
        aligned = proto[col]
        identity = torch.arange(K, device=proto.device)
        perm_like = (col != identity).float().mean().item()
        return aligned, perm_like

    # mode == "ot": soft coupling, anchor slot i = barycenter of client
    # prototypes weighted by the transport plan row T[i, :].
    T = _sinkhorn(cost, eps, iters)               # (K, K), row sums = 1/K
    aligned = (T @ proto) / T.sum(1, keepdim=True).clamp(min=1e-12)
    # Diagonal mass (scaled to [0,1] by *K) measures closeness to identity.
    diag_frac = (torch.diagonal(T) * K).mean().item()
    perm_like = 1.0 - diag_frac
    return aligned, perm_like


class Server:
    """Federated server managing global prototypes.

    The server performs FedAvg on prototypes uploaded by clients. Prototype
    aggregation is the ONLY information shared — model parameters (encoder,
    predictor) remain local.

    Prototypes are constrained to the unit hypersphere (each row L2-normalized)
    after aggregation. This is essential for numerical stability: the soft
    assignment uses cosine similarity, whose gradient w.r.t. a prototype scales
    as 1/‖p‖. With z_inv layer-normalized, the prototype magnitude is otherwise
    a free, ungradiented direction that random-walks; once a prototype's norm
    collapses toward 0 the cosine-similarity gradient explodes → NaN. Pinning
    ‖p‖ = 1 every round removes both the collapse and the inflation failure modes.

    Prototype indices carry no intrinsic meaning: client A's slot k and client
    B's slot k may specialize to different modes, so plain index-wise FedAvg can
    average unrelated prototypes together and collapse them. When ``align_mode``
    is "ot" or "hungarian", each client's prototypes are first aligned to this
    round's broadcast anchor (``self.global_prototypes``) before averaging.

    Args:
        K: number of global prototypes
        hid_dim: prototype dimension
        device: torch device
        align_mode: "none" | "ot" | "hungarian" — pre-aggregation alignment
        align_eps: Sinkhorn entropic regularization (ot only)
        align_iters: Sinkhorn iteration count (ot only)
    """

    def __init__(self, K, hid_dim, device="cpu",
                 align_mode="none", align_eps=0.05, align_iters=50,
                 normalize_codebook=True):
        self.K = K
        self.hid_dim = hid_dim
        self.device = device
        self.align_mode = align_mode
        self.align_eps = align_eps
        self.align_iters = align_iters
        self.normalize_codebook = normalize_codebook
        self.last_align_diag = None  # diagnostics from the most recent aggregate()

        # Global prototypes initialized on the unit hypersphere.
        proto_init = torch.randn(K, hid_dim, device=device)
        if self.normalize_codebook:
            self.global_prototypes = F.normalize(proto_init, dim=-1, eps=1e-6)
        else:
            self.global_prototypes = proto_init

    def aggregate(self, client_prototypes, weights=None):
        """FedAvg aggregation of client prototypes, projected to the unit sphere.

        When ``align_mode`` != "none", each client's prototypes are aligned to
        the current global anchor (``self.global_prototypes``, still the value
        broadcast this round) before the weighted sum.

        Args:
            client_prototypes: list of (K, hid_dim) tensors from each client
            weights: optional list of client weights (e.g., dataset sizes).
                     If None, uniform averaging.

        Returns:
            global_prototypes: aggregated (K, hid_dim) tensor, unit-normalized
        """
        if weights is None:
            weights = [1.0 / len(client_prototypes)] * len(client_prototypes)

        anchor = self.global_prototypes  # this round's broadcast = alignment ref
        aggregated = torch.zeros_like(self.global_prototypes)
        perm_likes, shifts = [], []
        for proto, w in zip(client_prototypes, weights):
            p = proto.to(self.device)
            if self.align_mode != "none":
                aligned, perm_like = _align_one(
                    anchor, p, self.align_mode, self.align_eps, self.align_iters
                )
                # Mean cosine displacement between raw and aligned prototypes.
                shift = (1.0 - F.cosine_similarity(
                    F.normalize(p, dim=-1, eps=1e-6),
                    F.normalize(aligned, dim=-1, eps=1e-6),
                    dim=-1,
                ).mean()).item()
                perm_likes.append(perm_like)
                shifts.append(shift)
                p = aligned
            aggregated += w * p

        if self.align_mode != "none" and perm_likes:
            self.last_align_diag = {
                "perm_frac": float(sum(perm_likes) / len(perm_likes)),
                "mean_shift": float(sum(shifts) / len(shifts)),
            }
        else:
            self.last_align_diag = None

        # Project back onto the unit hypersphere (spherical FedAvg).
        if self.normalize_codebook:
            self.global_prototypes = F.normalize(aggregated, dim=-1, eps=1e-6)
        else:
            self.global_prototypes = aggregated
        return self.global_prototypes

    def broadcast(self):
        """Return current global prototypes for distribution to clients."""
        return self.global_prototypes.clone()

    def aggregate_sparse(self, sparse_updates, weights=None):
        """Sparse prototype aggregation: only the submitted indices are updated.

        Each client uploads only its top-k dominant prototypes, identified by
        index. Global prototype *k* is averaged from the clients that chose to
        upload index *k*; prototypes with zero contributors retain their
        previous global value.

        Unlike ``aggregate()``, this method skips OT/Hungarian alignment:
        the indices themselves encode the alignment (each client's prototype
        started from the same global broadcast). The ``align_mode`` attribute
        is ignored.

        Args:
            sparse_updates: list of tuples (indices, proto_vecs) where
                ``indices`` is a 1-D int tensor of length topk and
                ``proto_vecs`` is a (topk, hid_dim) float tensor.
            weights: optional list of per-client scalar weights (e.g.
                     dataset-size fractions). If None, uniform.

        Returns:
            global_prototypes: aggregated (K, hid_dim) tensor, unit-normalized
        """
        K, D = self.global_prototypes.shape
        if weights is None:
            w = [1.0 / len(sparse_updates)] * len(sparse_updates)
        else:
            w = list(weights)

        # Accumulate weighted sum per prototype slot.
        acc = torch.zeros(K, D, device=self.device, dtype=self.global_prototypes.dtype)
        sum_w = torch.zeros(K, device=self.device)

        for (indices, proto_vecs), weight in zip(sparse_updates, w):
            indices = indices.to(self.device)
            proto_vecs = proto_vecs.to(self.device).to(self.global_prototypes.dtype)
            acc.index_add_(0, indices, weight * proto_vecs)
            w_t = torch.full_like(indices, weight, dtype=torch.float32)
            sum_w.index_add_(0, indices, w_t)

        # For every prototype slot with at least one contributor, use the
        # weighted average; for slots with no contributors, keep the previous
        # global value.
        has_contrib = sum_w > 0
        aggregated = self.global_prototypes.clone()
        aggregated[has_contrib] = (
            acc[has_contrib] / sum_w[has_contrib, None].clamp(min=1e-12)
        )
        if self.normalize_codebook:
            aggregated[has_contrib] = F.normalize(
                aggregated[has_contrib], dim=-1, eps=1e-6,
            )

        self.global_prototypes = aggregated
        self.last_align_diag = None
        return self.global_prototypes

    def get_prototypes(self):
        return self.global_prototypes
