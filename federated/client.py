"""Federated client: local training loop."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, TensorDataset

from utils.mi import gaussian_mi


def _add_codebook_noise(protos, noise_std, noise_type):
    """Add noise to prototype vectors, then re-normalize to the unit sphere.

    Args:
        protos: (..., hid_dim) tensor of prototype vectors
        noise_std: float, standard deviation / scale of the noise
        noise_type: "gaussian" | "laplace" | "uniform" | "rademacher"

    Returns:
        noisy_protos: same shape as protos, on the unit sphere
    """
    if noise_std <= 0:
        return protos
    shape = protos.shape
    if noise_type == "gaussian":
        noise = torch.randn(shape, device=protos.device, dtype=protos.dtype) * noise_std
    elif noise_type == "laplace":
        # Laplace(0, b) via inverse CDF: b * sign(U-0.5) * log(1 - 2|U-0.5|)
        u = torch.rand(shape, device=protos.device, dtype=protos.dtype) - 0.5
        noise = -noise_std * u.sign() * torch.log1p(-2 * u.abs())
    elif noise_type == "uniform":
        noise = (torch.rand(shape, device=protos.device, dtype=protos.dtype) * 2 - 1) * noise_std
    elif noise_type == "rademacher":
        noise = (torch.randint(0, 2, shape, device=protos.device).to(protos.dtype) * 2 - 1) * noise_std
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    return F.normalize(protos + noise, dim=-1, eps=1e-6)


class Client:
    """A federated client with local data and model.

    Args:
        client_id: unique client identifier
        model: CausalFedST model instance
        X_train, Y_train: local training data (numpy arrays)
        X_val, Y_val: local validation data (numpy arrays)
        batch_size: local batch size
        lr: learning rate
        local_epochs: number of local epochs per round
        device: torch device
    """

    def __init__(
        self,
        client_id,
        model,
        X_train,
        Y_train,
        T_train=None,
        X_val=None,
        Y_val=None,
        T_val=None,
        batch_size=32,
        lr=1e-3,
        local_epochs=5,
        device="cpu",
        weight_decay=1e-4,
        target_mean=None,
        target_std=None,
        mask_threshold=0.1,
        mi_weight=0.0,
        freeze_prototypes=False,
        inv_weight=0.0,
        freeze_encoder=False,
    ):
        self.client_id = client_id
        self.model = model
        self.device = device
        self.local_epochs = local_epochs
        self.freeze_prototypes = freeze_prototypes
        self.codebook_noise_std = 0.0
        self.codebook_noise_type = "gaussian"
        self.lr = lr
        self.weight_decay = weight_decay

        # Per-client normalization stats, used to recover the raw-scale target
        # so missing values (encoded as 0 mph in METR-LA) can be masked out of
        # the loss. When stats are None, no masking is applied.
        self.target_mean = target_mean
        self.target_std = target_std
        self.mask_threshold = mask_threshold
        self.mi_weight = mi_weight
        self.inv_weight = inv_weight
        self.fedprox_mu = 0.0
        self.ditto_lambda = 0.0
        self.global_model = None
        self.global_optimizer = None

        # NB: the model is left on its current device (typically CPU) here. The
        # training loop (federated_train's GPU assignment, or VectorGroup in the
        # vectorized path) moves every model to its target GPU exactly once,
        # before any forward pass. Moving eagerly here would be redundant work
        # (and a double move for multi-GPU), which dominates client construction
        # time at scale. The Adam state is created lazily on the first step, on
        # whatever device the params then live on, so this is semantics-preserving.

        # Create data loaders
        train_tensors = [torch.from_numpy(X_train), torch.from_numpy(Y_train)]
        if T_train is not None:
            train_tensors.append(torch.from_numpy(T_train))
        self.train_loader = DataLoader(
            TensorDataset(*train_tensors),
            batch_size=batch_size,
            shuffle=True,
        )
        self.val_loader = None
        if X_val is not None and Y_val is not None:
            val_tensors = [torch.from_numpy(X_val), torch.from_numpy(Y_val)]
            if T_val is not None:
                val_tensors.append(torch.from_numpy(T_val))
            self.val_loader = DataLoader(
                TensorDataset(*val_tensors),
                batch_size=batch_size,
                shuffle=False,
            )

        # Transfer / two-stage: hold the loaded codebook (and optionally the
        # invariant encoder) fixed and optimize only the remaining parameters.
        if freeze_prototypes:
            self.model.freeze_prototypes()
        if freeze_encoder:
            self.model.freeze_invariant_path()
        opt_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            opt_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    def enable_ditto(self, global_model, ditto_lambda):
        """Attach Ditto global/personalized model pair to this client."""
        self.global_model = copy.deepcopy(global_model)
        self.global_model.load_state_dict(self.model.state_dict())
        self.ditto_lambda = float(ditto_lambda)
        self.global_optimizer = torch.optim.Adam(
            [p for p in self.global_model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def rebuild_optimizer(self):
        """Recreate the optimizer after changing per-client trainable params."""
        opt_params = [p for p in self.model.parameters() if p.requires_grad]
        if not opt_params:
            raise ValueError(f"client {self.client_id} has no trainable parameters")
        self.optimizer = torch.optim.Adam(
            opt_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _masked_mae(self, y_hat, y):
        """Masked MAE loss, ignoring missing values.

        y_hat, y are in normalized space with shape (B, horizon, 1). When
        per-client normalization stats are available, targets whose raw value
        is below ``mask_threshold`` (missing entries, encoded as 0 in METR-LA)
        are excluded. Otherwise all entries are kept.
        """
        if self.target_mean is not None and self.target_std is not None:
            y_raw = y * self.target_std + self.target_mean
            mask = (y_raw > self.mask_threshold).float()
        else:
            mask = torch.ones_like(y)
        err = torch.abs(y_hat - y) * mask
        denom = mask.sum().clamp(min=1.0)
        return err.sum() / denom

    def train_local(self):
        """Run local training for local_epochs.

        Returns:
            train_loss: average training loss (includes MI regulariser when
                        mi_weight > 0)
            prototypes: updated local prototypes (detached)
        """
        self.model.train()
        if all(not p.requires_grad for p in self.model.encoder.parameters()):
            self.model.encoder.eval()
        use_mi = self.mi_weight > 0 and self.model.use_residual
        use_aux = self.inv_weight > 0 and getattr(self.model, "use_aux_inv", False)
        total_loss = 0.0
        total_mi = 0.0
        n_batches = 0

        prox_ref = None
        if getattr(self, "fedprox_mu", 0.0) > 0:
            prox_ref = {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()
                if torch.is_floating_point(v)
            }

        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                batch_x, batch_y = batch[:2]
                batch_t = batch[2] if len(batch) > 2 else None
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if batch_t is not None:
                    batch_t = batch_t.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                if use_aux:
                    y_hat, y_hat_inv, _, _ = self.model(
                        batch_x, temporal_id=batch_t, return_inv=True
                    )
                    loss_mae = self._masked_mae(y_hat, batch_y[..., :1])
                    loss_inv = self._masked_mae(y_hat_inv, batch_y[..., :1])
                    loss = loss_mae + self.inv_weight * loss_inv
                    mi = torch.tensor(0.0)
                elif use_mi:
                    y_hat, z_inv, z_res, _ = self.model(
                        batch_x, temporal_id=batch_t, return_residual=True
                    )
                    loss_mae = self._masked_mae(y_hat, batch_y[..., :1])
                    # Squeeze T=1 dim: (B, 1, D) → (B, D)
                    mi = gaussian_mi(z_inv.squeeze(1), z_res.squeeze(1))
                    loss = loss_mae + self.mi_weight * mi
                else:
                    y_hat, _, _ = self.model(batch_x, temporal_id=batch_t)
                    loss = self._masked_mae(y_hat, batch_y[..., :1])
                    mi = torch.tensor(0.0)
                if prox_ref is not None:
                    prox = torch.zeros((), device=self.device)
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and name in prox_ref:
                            prox = prox + (param - prox_ref[name].to(self.device)).pow(2).sum()
                    loss = loss + 0.5 * self.fedprox_mu * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                self.model.normalize_prototypes_()  # keep ‖prototype‖ = 1

                total_loss += loss.item()
                total_mi += mi.item() if isinstance(mi, torch.Tensor) else mi
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        prototypes = self.model.get_prototypes()
        prototypes = _add_codebook_noise(prototypes, self.codebook_noise_std, self.codebook_noise_type)
        return avg_loss, prototypes

    def train_local_sparse(self, proto_topk=1):
        """Run local training and return only the top-k dominant prototypes.

        After the usual local-epoch loop, one extra no-grad pass over the
        training data computes the average softmax assignment weights and
        selects the top-k prototypes. Only those k prototypes (and their
        indices) are returned, reducing upload communication by ~K/topk.

        Args:
            proto_topk: number of dominant prototypes to upload (default 1).

        Returns:
            avg_loss: float — average training loss over the last epoch.
            topk_indices: LongTensor (topk,) — indices of the dominant prototypes.
            topk_protos: FloatTensor (topk, hid_dim) — the dominant prototype vectors.
        """
        # Standard local training (same as train_local).
        self.model.train()
        if all(not p.requires_grad for p in self.model.encoder.parameters()):
            self.model.encoder.eval()
        use_mi = self.mi_weight > 0 and self.model.use_residual
        use_aux = self.inv_weight > 0 and getattr(self.model, "use_aux_inv", False)
        total_loss = 0.0
        n_batches = 0

        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                batch_x, batch_y = batch[:2]
                batch_t = batch[2] if len(batch) > 2 else None
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if batch_t is not None:
                    batch_t = batch_t.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                if use_aux:
                    y_hat, y_hat_inv, _, _ = self.model(
                        batch_x, temporal_id=batch_t, return_inv=True)
                    loss = (self._masked_mae(y_hat, batch_y[..., :1])
                            + self.inv_weight * self._masked_mae(y_hat_inv, batch_y[..., :1]))
                elif use_mi:
                    y_hat, _, _, _ = self.model(
                        batch_x, temporal_id=batch_t, return_residual=True)
                    loss = self._masked_mae(y_hat, batch_y[..., :1])
                else:
                    y_hat, _, _ = self.model(batch_x, temporal_id=batch_t)
                    loss = self._masked_mae(y_hat, batch_y[..., :1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                self.model.normalize_prototypes_()

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Determine dominant prototypes from average assignment weights.
        avg_weights = self.model.compute_assignment_weights(self.train_loader)
        _, topk_indices = torch.topk(avg_weights, min(proto_topk, len(avg_weights)))
        full_protos = self.model.get_prototypes()
        full_protos = _add_codebook_noise(full_protos, self.codebook_noise_std, self.codebook_noise_type)
        topk_protos = full_protos[topk_indices].clone()

        return avg_loss, topk_indices, topk_protos

    def train_ditto(self):
        """Ditto update: train a global model, then a personalized model."""
        if self.global_model is None or self.global_optimizer is None:
            self.enable_ditto(self.model, self.ditto_lambda)
        self.global_model.to(self.device)
        self.global_model.train()
        total_loss = 0.0
        n_batches = 0

        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                batch_x, batch_y = batch[:2]
                batch_t = batch[2] if len(batch) > 2 else None
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if batch_t is not None:
                    batch_t = batch_t.to(self.device)
                self.global_optimizer.zero_grad(set_to_none=True)
                y_hat, _, _ = self.global_model(batch_x, temporal_id=batch_t)
                loss = self._masked_mae(y_hat, batch_y[..., :1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=5.0)
                self.global_optimizer.step()
                self.global_model.normalize_prototypes_()
                total_loss += loss.item()
                n_batches += 1

        global_ref = {
            k: v.detach().clone()
            for k, v in self.global_model.state_dict().items()
            if torch.is_floating_point(v)
        }
        self.model.train()
        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                batch_x, batch_y = batch[:2]
                batch_t = batch[2] if len(batch) > 2 else None
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if batch_t is not None:
                    batch_t = batch_t.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                y_hat, _, _ = self.model(batch_x, temporal_id=batch_t)
                loss = self._masked_mae(y_hat, batch_y[..., :1])
                prox = torch.zeros((), device=self.device)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in global_ref:
                        prox = prox + (param - global_ref[name].to(self.device)).pow(2).sum()
                loss = loss + 0.5 * self.ditto_lambda * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                self.model.normalize_prototypes_()

        protos = self.global_model.get_prototypes()
        protos = _add_codebook_noise(protos, self.codebook_noise_std, self.codebook_noise_type)
        return total_loss / max(n_batches, 1), protos

    def evaluate(self, loader="val"):
        """Evaluate model on validation or test data.

        Returns:
            avg_loss: average masked MAE loss
        """
        loader = self.val_loader if loader == "val" else self.test_loader
        if loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                batch_x, batch_y = batch[:2]
                batch_t = batch[2] if len(batch) > 2 else None
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if batch_t is not None:
                    batch_t = batch_t.to(self.device)
                y_hat, _, _ = self.model(batch_x, temporal_id=batch_t)
                loss = self._masked_mae(y_hat, batch_y[..., :1])
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate_raw_mae(self, loader="val"):
        """Raw-scale MAE for this client plus metadata for worst-client logs."""
        loader = self.val_loader if loader == "val" else self.test_loader
        if loader is None:
            return {
                "client_id": self.client_id,
                "raw_mae": float("inf"),
                "target_std": float(self.target_std) if self.target_std is not None else float("nan"),
            }

        self.model.eval()
        se = 0.0
        cnt = 0.0
        with torch.no_grad():
            for batch in loader:
                batch_x, batch_y = batch[:2]
                batch_t = batch[2] if len(batch) > 2 else None
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if batch_t is not None:
                    batch_t = batch_t.to(self.device)
                y_hat, _, _ = self.model(batch_x, temporal_id=batch_t)

                if self.target_mean is not None and self.target_std is not None:
                    y_true = batch_y[..., :1] * self.target_std + self.target_mean
                    y_pred = y_hat * self.target_std + self.target_mean
                    mask = y_true > self.mask_threshold
                else:
                    y_true = batch_y[..., :1]
                    y_pred = y_hat
                    mask = torch.ones_like(y_true, dtype=torch.bool)
                se += torch.abs(y_pred[mask] - y_true[mask]).sum().item()
                cnt += mask.sum().item()

        return {
            "client_id": self.client_id,
            "raw_mae": se / max(cnt, 1.0),
            "target_std": float(self.target_std) if self.target_std is not None else float("nan"),
        }

    def set_prototypes(self, prototypes):
        """Receive global prototypes from server."""
        self.model.set_prototypes(prototypes)
