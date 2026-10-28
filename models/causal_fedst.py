"""
CausalFedST: Causal Federated Spatio-Temporal Forecasting.

Architecture: X → Encoder(CaST backbone) → PrototypeMatching → Predictor → Ŷ
Loss: MSE(Ŷ, Y)

The causal disentanglement is structural: global prototypes are shared
and Federated-averaged across clients, so only cross-client invariant
causal dynamics survive in the prototypes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_encoder
from .prototype import PrototypeMatching
from .predictor import Predictor


class CausalFedST(nn.Module):
    """Causal Federated Spatio-Temporal forecasting model.

    Args:
        input_dim: input feature dimension (default 1)
        hid_dim: hidden dimension throughout the model
        seq_len: input sequence length (window size)
        horizon: prediction horizon (output steps)
        K: number of global prototypes (causal environment types)
        tau: temperature for prototype soft assignment
        conv_depth: number of dilated conv blocks in encoder
        attn_heads: number of attention heads in self-attention
        dropout: dropout rate
    """

    def __init__(
        self,
        input_dim=1,
        hid_dim=64,
        seq_len=12,
        horizon=3,
        K=10,
        tau=0.1,
        stid_num_layers=3,
        dropout=0.1,
        use_residual=True,
        residual_mode="learnable",
        residual_scale=0.25,
        use_residual_gate=None,
        residual_gate_init=-2.0,
        encoder_type="stid",
        encoder_kwargs=None,
        use_prototype=True,
        use_aux_inv=False,
        separate_residual=False,
        normalize_codebook=True,
        prototype_init="randn",
        hard_proto=False,
        multivariate=False,
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.K = K
        self.tau = tau
        # When the prototype path is disabled (e.g. fl_mode local/fedavg), the
        # encoder output feeds the predictor directly; residual splitting (which
        # is defined as z - z_inv) is meaningless, so force it off.
        self.use_prototype = bool(use_prototype)
        if not self.use_prototype:
            use_residual = False
        self.use_residual = use_residual
        if use_residual_gate is not None:
            residual_mode = "learnable" if use_residual_gate else "none"
        if residual_mode not in {"none", "fixed", "learnable", "detach", "detach_fixed"}:
            raise ValueError(
                "residual_mode must be one of: none, fixed, learnable, detach, detach_fixed"
            )
        self.residual_mode = residual_mode if use_residual else "none"
        self.residual_scale = float(residual_scale)

        self.encoder = build_encoder(
            encoder_type,
            input_dim=input_dim,
            hid_dim=hid_dim,
            seq_len=seq_len,
            num_layers=stid_num_layers,
            dropout=dropout,
            encoder_kwargs=encoder_kwargs,
            multivariate=multivariate,
        )

        self.prototype = PrototypeMatching(
            hid_dim=hid_dim,
            K=K,
            tau=tau,
            normalize_codebook=normalize_codebook,
            prototype_init=prototype_init,
            hard=hard_proto,
        )

        self.predictor = Predictor(
            hid_dim=hid_dim,
            horizon=horizon,
            use_residual=use_residual,
            dropout=dropout,
        )

        # Auxiliary invariant head: predicts y from z_inv alone. With an explicit
        # supervised loss on this head (see --aux_inv_weight), the invariant
        # predictive signal is forced through the shared codebook channel rather
        # than being bypassed by the local residual z_res. Local (never shared),
        # and only meaningful when the prototype path is active.
        self.use_aux_inv = bool(use_aux_inv) and self.use_prototype
        if self.use_aux_inv:
            self.inv_head = Predictor(
                hid_dim=hid_dim,
                horizon=horizon,
                use_residual=False,
                dropout=dropout,
            )
        else:
            self.inv_head = None

        # Optional separate residual encoder: z_res = phi_res(x) from an
        # independent backbone, instead of the arithmetic complement z - z_inv.
        # Used by two-stage training, where the invariant path (encoder + Q) is
        # frozen and only this residual branch + predictor are trained.
        self.separate_residual = bool(separate_residual) and self.use_residual
        if self.separate_residual:
            self.residual_encoder = build_encoder(
                encoder_type,
                input_dim=input_dim,
                hid_dim=hid_dim,
                seq_len=seq_len,
                num_layers=stid_num_layers,
                dropout=dropout,
                encoder_kwargs=encoder_kwargs,
                multivariate=multivariate,
            )
        else:
            self.residual_encoder = None
        if self.residual_mode == "learnable":
            self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        else:
            self.register_parameter("residual_gate", None)

    def forward(self, x, temporal_id=None, return_residual=False, return_inv=False):
        """Forward pass.

        Args:
            x: (B, T, input_dim) input time series
            temporal_id: optional (B, 2) [time_of_day, day_of_week]
            return_residual: if True, also return the raw residual
                z_res_raw = z − z_inv (before any gating / scaling),
                for use in independence-regularisation losses.

        Returns:
            When return_residual is False (default):
                y_hat: (B, horizon, 1) predictions
                z_inv: (B, T, hid_dim) invariant representation
                weights: (B, T, K) prototype assignment weights
            When return_residual is True:
                y_hat, z_inv, z_res_raw, weights
                where z_res_raw = z − z_inv (before gating / scaling).
        """
        z = self.encoder(x, temporal_id=temporal_id)
        if self.use_prototype:
            z_inv, weights = self.prototype(z)
        else:
            z_inv, weights = z, None
        z_res_raw = z - z_inv if (self.use_residual and return_residual) else None
        if self.use_residual and self.separate_residual:
            # Independent residual branch: z_res from its own encoder.
            z_res = self.residual_encoder(x, temporal_id=temporal_id)
        elif self.use_residual:
            z_res = z - z_inv
            if self.residual_mode in {"detach", "detach_fixed"}:
                z_res = z_res.detach()
            if self.residual_mode == "learnable":
                z_res = torch.sigmoid(self.residual_gate) * z_res
            elif self.residual_mode in {"fixed", "detach_fixed"}:
                z_res = self.residual_scale * z_res
        else:
            z_res = None
        y_hat = self.predictor(z_inv, z_res)
        if return_inv:
            # z_inv-only readout (the de-confounded / transferable predictor).
            y_hat_inv = self.inv_head(z_inv) if self.inv_head is not None else None
            return y_hat, y_hat_inv, z_inv, weights
        if return_residual:
            return y_hat, z_inv, z_res_raw, weights
        return y_hat, z_inv, weights

    def get_prototypes(self):
        """Get prototypes for server aggregation."""
        return self.prototype.get_prototypes()

    def set_prototypes(self, prototypes):
        """Set prototypes from server aggregation."""
        self.prototype.set_prototypes(prototypes)

    def normalize_prototypes_(self):
        """Project prototypes onto the unit hypersphere (in place)."""
        self.prototype.normalize_prototypes_()

    def get_trainable_params(self):
        """Get trainable parameters excluding prototypes (for client-side optimizer)."""
        return [
            p for name, p in self.named_parameters()
            if "prototypes" not in name
        ]

    @torch.no_grad()
    def compute_assignment_weights(self, loader, proxy_prototypes=None):
        """Compute average softmax assignment weights over a data loader.

        Args:
            loader: DataLoader yielding (x,) or (x, temporal_id) batches.
            proxy_prototypes: optional (K, D) prototype tensor. If None, uses
                the model's own current prototypes.

        Returns:
            avg_weights: (K,) float tensor — average softmax weights over all
                         samples in the loader.
        """
        was_training = self.training
        self.eval()
        K = self.prototype.K
        tau = self.prototype.tau
        prototypes = (F.normalize(proxy_prototypes, dim=-1, eps=1e-6)
                      if proxy_prototypes is not None
                      else F.normalize(self.prototype.prototypes, dim=-1, eps=1e-6))

        device = next(self.parameters()).device
        sum_weights = torch.zeros(K)
        n_samples = 0
        for batch in loader:
            x = batch[0].to(device)
            t = batch[2].to(device) if len(batch) > 2 else None
            z = self.encoder(x, temporal_id=t)
            B, T, D = z.shape
            z_flat = z.reshape(B * T, D)
            z_norm = F.normalize(z_flat, dim=-1)
            sim = z_norm @ prototypes.T
            w = F.softmax(sim / tau, dim=-1)  # (B*T, K)
            sum_weights = sum_weights + w.sum(dim=0).cpu()
            n_samples += w.shape[0]

        if was_training:
            self.train()
        return sum_weights / max(n_samples, 1)

    def freeze_invariant_path(self):
        """Freeze the invariant path (encoder + codebook) for stage-2 training.

        After a stage-1 z_inv-only run has shaped z_inv into a meaningful
        invariant, freezing the encoder and prototypes pins z_inv; only the
        separate residual branch + predictor then train, so z_inv cannot drift
        as the residual restores in-distribution accuracy.
        """
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.prototype.prototypes.requires_grad_(False)

    def freeze_prototypes(self):
        """Freeze the prototype codebook (no gradient updates).

        Used by transfer experiments: a pretrained global codebook Q is loaded
        and held fixed while only the local backbone+head are trained. The
        cosine assignment uses prototype directions only and Q is already on the
        unit sphere, so no further normalization is required.
        """
        self.prototype.prototypes.requires_grad_(False)
