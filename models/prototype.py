"""
Prototype matching via soft assignment.

Each prototype p_i is a learnable vector in the representation space.
Soft assignment: w_i = softmax(cosine_sim(z, p_i) / tau)
Output: z_inv = Σ w_i * p_i  (weighted sum of prototypes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMatching(nn.Module):
    """Soft prototype matching layer.

    Given encoder output Z_k, compute attention weights over K global
    prototypes and produce Z_inv_k as their weighted sum.

    Args:
        hid_dim: dimension of prototypes and encoder output
        K: number of global prototypes (causal environment types)
        tau: temperature for softmax (lower = harder assignment)
    """

    def __init__(self, hid_dim, K, tau=0.1, normalize_codebook=True,
                 prototype_init="randn", hard=False):
        super().__init__()
        self.K = K
        self.tau = tau
        self.normalize_codebook = normalize_codebook
        self.prototype_init = prototype_init
        self.hard = hard  # True → VQ-style argmax + straight-through

        if prototype_init == "cast":
            # CaST-style: compact uniform init near the origin.
            # Prototypes start as tiny vectors clustered around zero and
            # spread during training (hard-VQ + no norm constraint).
            self.prototypes = nn.Parameter(
                torch.empty(K, hid_dim).uniform_(-1.0 / K, 1.0 / K)
            )
        elif prototype_init == "randn":
            # Default: isotropic Gaussian → uniform on the unit sphere after
            # normalisation (good spread from the start).
            self.prototypes = nn.Parameter(
                torch.randn(K, hid_dim) * 0.1
            )
        else:
            raise ValueError(
                f"Unknown prototype_init={prototype_init}; expected 'randn' or 'cast'"
            )

    def forward(self, z):
        """Match encoder output to prototypes.

        Args:
            z: (B, T, hid_dim) encoder output

        Returns:
            z_inv: (B, T, hid_dim) invariant representation
            weights: (B, T, K) soft assignment weights
        """
        B, T, D = z.shape

        # Cosine similarity between each (batch, time) embedding and each prototype
        z_flat = z.reshape(B * T, D)  # (B*T, D)
        z_norm = F.normalize(z_flat, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim = z_norm @ p_norm.T  # (B*T, K)

        if self.hard:
            # ── VQ-style hard assignment with soft gradient ──
            # Forward:  z_inv = p_k  (closest prototype, hard)
            # Backward: gradient through softmax @ p → prototypes + encoder
            # All ops vmap-compatible (no scatter_ / gumbel_softmax).
            k = sim.argmax(dim=-1)                         # (B*T,) hard idx
            weights_soft = F.softmax(sim / self.tau, dim=-1)  # (B*T,K) for grad
            z_inv_hard = F.embedding(k, p_norm)            # (B*T, D)
            z_inv_soft = weights_soft @ p_norm             # (B*T, D) differentiable
            # STE: forward = hard, backward = soft gradient
            z_inv_flat = z_inv_hard + (z_inv_soft - z_inv_hard).detach()
            weights = F.one_hot(k, num_classes=self.K).float()   # (B*T, K)
            weights = weights.reshape(B, T, self.K)
            z_inv = z_inv_flat.reshape(B, T, D)
            return z_inv, weights

        # Soft assignment (original path)
        weights = F.softmax(sim / self.tau, dim=-1)  # (B*T, K)
        weights = weights.reshape(B, T, self.K)

        # Weighted sum of prototypes
        z_inv = weights @ self.prototypes  # (B*T, D) → (B, T, D)

        return z_inv, weights

    def get_prototypes(self):
        """Return prototype vectors (detached, for server aggregation)."""
        return self.prototypes.data.clone()

    def set_prototypes(self, prototypes):
        """Set prototype vectors (from server aggregation)."""
        self.prototypes.data.copy_(prototypes.to(self.prototypes.device))

    @torch.no_grad()
    def normalize_prototypes_(self):
        """Project prototypes back onto the unit hypersphere (in place).

        Called after every optimizer step: the soft-assignment cosine similarity
        has a 1/‖p‖ gradient singularity, so a prototype whose norm drifts toward
        zero blows up. Pinning ‖p‖ = 1 at all times removes that failure mode (and
        the dual scale-inflation one). Output is unaffected since z_inv is
        layer-normalized and the assignment uses only prototype direction.

        When normalize_codebook=False, this is a no-op — prototype norms are
        free to drift (e.g., for ablation experiments).
        """
        if not self.normalize_codebook:
            return
        self.prototypes.data.copy_(
            F.normalize(self.prototypes.data, dim=-1, eps=1e-6)
        )