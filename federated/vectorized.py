"""Vectorized (vmap) ensemble training for clients sharing one GPU.

All clients have an identical model architecture, so the ~N clients pinned to a
single GPU can be processed as one batched ("ensemble") computation instead of a
Python for-loop over clients. This turns N sequential small kernels into one big
kernel per step, saturating the GPU and removing the within-GPU serialization.

Mechanism (PyTorch torch.func ensembling pattern):
    params, buffers = stack_module_state([client models])   # leaves: (M, *shape)
    y = vmap(functional_call(base, params, x))(...)          # one call, all clients

Numerical semantics are kept identical to the per-client loop:
    * loss is the SUM of per-client masked MAE, so each client's gradient equals
      its own dL_m/dθ_m (params are independent across the client dim);
    * gradient clipping is computed per client (joint norm over that client's
      params), matching torch.nn.utils.clip_grad_norm_ applied per client;
    * Adam over the stacked params updates each client elementwise == per-client
      Adam;
    * prototypes are broadcast in-place each round (copy_) and read back for
      FedAvg, exactly like Client.set_prototypes / get_prototypes.

Constraint: clients in a group must have aligned shapes (equal sample counts).
METR-LA satisfies this (all nodes share the same timeline). fraction<1 client
sampling is not supported on this path — all clients train every round.
"""

import copy

import torch
import torch.nn.functional as F
from torch import vmap
from torch.func import functional_call, stack_module_state

from utils.mi import gaussian_mi_per_client
from federated.client import _add_codebook_noise

PROTO_KEY = "prototype.prototypes"


class VectorGroup:
    """A vmapped ensemble of identical-architecture client models on one device."""

    def __init__(self, clients, device, lr, weight_decay, mask_threshold=0.1,
                 max_grad_norm=5.0, mi_weight=0.0, freeze_prototypes=False,
                 inv_weight=0.0, freeze_encoder=False, fedprox_mu=0.0,
                 normalize_codebook=True, codebook_noise_std=0.0,
                 codebook_noise_type="gaussian"):
        self.clients = clients
        self.device = device
        self.M = len(clients)
        self.mask_threshold = mask_threshold
        self.max_grad_norm = max_grad_norm
        self.mi_weight = mi_weight
        self.freeze_prototypes = freeze_prototypes
        self.inv_weight = inv_weight
        self.freeze_encoder = freeze_encoder
        self.fedprox_mu = fedprox_mu
        self.normalize_codebook = normalize_codebook
        self.codebook_noise_std = codebook_noise_std
        self.codebook_noise_type = codebook_noise_type

        models = [c.model.to(device) for c in clients]
        # Skeleton on the meta device: functional_call swaps in real params, so the
        # meta module is never used for compute — only for its forward structure.
        self.base = copy.deepcopy(models[0]).to("meta")

        # Frozen params stay as no-grad leaves (functional_call still reads them)
        # but are excluded from the optimizer. Covers transfer (codebook fixed)
        # and two-stage (invariant encoder + codebook fixed).
        def _is_frozen(k):
            if freeze_prototypes and k == PROTO_KEY:
                return True
            if freeze_encoder and (k.startswith("encoder.") or k == PROTO_KEY):
                return True
            return False

        params, buffers = stack_module_state(models)
        self.params = {
            k: v.detach().clone().requires_grad_(not _is_frozen(k))
            for k, v in params.items()
        }
        self.buffers = {k: v.detach().clone() for k, v in buffers.items()}

        opt_params = [v for k, v in self.params.items() if not _is_frozen(k)]
        self.opt = torch.optim.Adam(
            opt_params, lr=lr, weight_decay=weight_decay
        )

        # Stack per-client data on the device once: inputs (M, N, T, D),
        # targets (M, N, horizon, 1), optional temporal ids (M, N, 2).
        self.Xtr, self.Ytr, self.Ttr = self._stack_loader(clients, "train")
        self.Xval, self.Yval, self.Tval = self._stack_loader(clients, "val")

        # Per-client normalization stats for raw-scale masking: (M, 1, 1, 1).
        mean = torch.tensor([c.target_mean for c in clients], dtype=torch.float32)
        std = torch.tensor([c.target_std for c in clients], dtype=torch.float32)
        self.mean = mean.view(self.M, 1, 1, 1).to(device)
        self.std = std.view(self.M, 1, 1, 1).to(device)

    # ── data ──
    def _stack_loader(self, clients, which):
        xs, ys, ts = [], [], []
        has_temporal = None
        for c in clients:
            loader = c.train_loader if which == "train" else c.val_loader
            tensors = loader.dataset.tensors
            X, Y = tensors[:2]
            xs.append(X)
            ys.append(Y[..., :1])
            cur_has_temporal = len(tensors) > 2
            if has_temporal is None:
                has_temporal = cur_has_temporal
            elif has_temporal != cur_has_temporal:
                raise ValueError("All clients in a VectorGroup must agree on temporal-id tensors")
            if cur_has_temporal:
                ts.append(tensors[2])
        X = torch.stack(xs, dim=0).to(self.device)        # (M, N, T, D)
        Y = torch.stack(ys, dim=0).to(self.device)        # (M, N, horizon, 1)
        T = torch.stack(ts, dim=0).to(self.device) if has_temporal else None
        return X, Y, T

    # ── functional forward ──
    def _fwd(self, params, buffers, x, temporal_id=None):
        y_hat, _, _ = functional_call(
            self.base, (params, buffers), (x,), {"temporal_id": temporal_id}
        )
        return y_hat

    def _fwd_with_residual(self, params, buffers, x, temporal_id=None):
        y_hat, z_inv, z_res, _ = functional_call(
            self.base, (params, buffers), (x,),
            {"temporal_id": temporal_id, "return_residual": True},
        )
        return y_hat, z_inv, z_res

    def _fwd_with_inv(self, params, buffers, x, temporal_id=None):
        y_hat, y_hat_inv, _, _ = functional_call(
            self.base, (params, buffers), (x,),
            {"temporal_id": temporal_id, "return_inv": True},
        )
        return y_hat, y_hat_inv

    def _masked_mae_per_client(self, y_hat, y):
        """Per-client masked MAE → (M,)."""
        y_raw = y * self.std + self.mean
        mask = (y_raw > self.mask_threshold).float()
        err = (y_hat - y).abs() * mask
        denom = mask.flatten(1).sum(1).clamp(min=1.0)
        return err.flatten(1).sum(1) / denom

    def _clip_grad_per_client(self):
        """Reproduce per-client clip_grad_norm_(max_norm) over the stacked grads."""
        sq = torch.zeros(self.M, device=self.device)
        for p in self.params.values():
            if p.grad is not None:
                grad_sq = p.grad.detach().pow(2)
                if grad_sq.ndim == 1:
                    sq += grad_sq
                else:
                    sq += grad_sq.flatten(1).sum(1)
        scale = (self.max_grad_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)  # (M,)
        for p in self.params.values():
            if p.grad is not None:
                view = [self.M] + [1] * (p.grad.ndim - 1)
                p.grad.mul_(scale.view(view))

    # ── prototypes ──
    def set_global_prototype(self, proto):
        """Broadcast (K, D) global prototypes into every client's slice, in place."""
        with torch.no_grad():
            self.params[PROTO_KEY].copy_(
                proto.to(self.device).unsqueeze(0).expand(self.M, -1, -1)
            )

    def get_client_prototypes(self):
        """Per-client prototypes after local training → (M, K, D) on CPU."""
        protos = self.params[PROTO_KEY].detach().cpu()  # (M, K, D)
        protos = _add_codebook_noise(protos, self.codebook_noise_std, self.codebook_noise_type)
        return protos

    def client_sizes(self):
        return [self.Xtr.shape[1]] * self.M

    def fedavg_param_keys(self, key_pred, weights=None):
        """In-place weighted FedAvg across the client dim for matching keys.

        For every stacked param whose name matches ``key_pred``, replace each
        client's slice with the (size-weighted) mean over clients — i.e. all
        clients share one global value for that param. Mirrors how
        ``set_global_prototype`` overwrites the prototype slice each round.
        """
        if weights is None:
            w = torch.full((self.M,), 1.0 / self.M, device=self.device)
        else:
            w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            w = w / w.sum()
        with torch.no_grad():
            for k, v in self.params.items():
                if not key_pred(k):
                    continue
                wshape = (self.M,) + (1,) * (v.dim() - 1)
                mean = (v * w.view(wshape)).sum(dim=0, keepdim=True)  # (1, *shape)
                v.copy_(mean.expand_as(v))

    # ── train / eval ──
    def train_local_epochs(self, local_epochs, batch_size):
        self.base.train()
        use_mi = self.mi_weight > 0 and getattr(self.base, "use_residual", False)
        use_aux = self.inv_weight > 0 and getattr(self.base, "use_aux_inv", False)
        if use_aux:
            vfwd = vmap(self._fwd_with_inv, randomness="different")
        elif use_mi:
            vfwd = vmap(self._fwd_with_residual, randomness="different")
        else:
            vfwd = vmap(self._fwd, randomness="different")
        X, Y, Tids = self.Xtr, self.Ytr, self.Ttr
        N = X.shape[1]
        total, nb = 0.0, 0
        # FedProx: proximal reference = the round-start (post-broadcast) params.
        prox_ref = None
        if self.fedprox_mu > 0:
            prox_ref = {k: v.detach().clone()
                        for k, v in self.params.items() if v.requires_grad}
        for _ in range(local_epochs):
            perm = torch.argsort(torch.rand(self.M, N, device=self.device), dim=1)
            for b in range(0, N, batch_size):
                idx = perm[:, b:b + batch_size]                       # (M, Bcur)
                xb = torch.gather(
                    X, 1, idx[..., None, None].expand(-1, -1, X.shape[2], X.shape[3]))
                yb = torch.gather(
                    Y, 1, idx[..., None, None].expand(-1, -1, Y.shape[2], Y.shape[3]))
                tb = None
                if Tids is not None:
                    tb = torch.gather(
                        Tids, 1, idx[..., None].expand(-1, -1, Tids.shape[2]))

                self.opt.zero_grad(set_to_none=True)
                if use_aux:
                    y_hat, y_hat_inv = vfwd(self.params, self.buffers, xb, tb)
                    loss_mae = self._masked_mae_per_client(y_hat, yb)        # (M,)
                    loss_inv = self._masked_mae_per_client(y_hat_inv, yb)    # (M,)
                    loss_per = loss_mae + self.inv_weight * loss_inv         # (M,)
                elif use_mi:
                    y_hat, z_inv, z_res = vfwd(self.params, self.buffers, xb, tb)
                    # y_hat:  (M, Bcur, H, 1)
                    # z_inv:  (M, Bcur, 1, D) → squeeze T=1
                    # z_res:  (M, Bcur, 1, D)
                    loss_mae = self._masked_mae_per_client(y_hat, yb)  # (M,)
                    # .contiguous() materialises vmap's internal batched-
                    # tensor representation, otherwise torch.linalg ops
                    # fail with "lazy wrapper should be called at most once"
                    mi_per = gaussian_mi_per_client(
                        z_inv.squeeze(2).contiguous(),
                        z_res.squeeze(2).contiguous(),
                    )                                                   # (M,)
                    loss_per = loss_mae + self.mi_weight * mi_per      # (M,)
                else:
                    y_hat = vfwd(self.params, self.buffers, xb, tb)   # (M, Bcur, H, 1)
                    loss_per = self._masked_mae_per_client(y_hat, yb) # (M,)
                if prox_ref is not None:
                    # FedProx proximal term, computed per client over the
                    # stacked param dim: (mu/2) * ||theta_m - theta_ref_m||^2.
                    prox_per = torch.zeros(self.M, device=self.device)
                    for k, ref in prox_ref.items():
                        diff_sq = (self.params[k] - ref).pow(2)
                        if diff_sq.ndim == 1:
                            prox_per = prox_per + diff_sq
                        else:
                            prox_per = prox_per + diff_sq.flatten(1).sum(1)
                    loss_per = loss_per + 0.5 * self.fedprox_mu * prox_per
                loss_per.sum().backward()                            # per-client grads
                self._clip_grad_per_client()
                self.opt.step()
                # Project every client's prototypes back to the unit sphere,
                # keeping ‖p‖ = 1 within the round (not just at aggregation).
                if self.normalize_codebook:
                    with torch.no_grad():
                        p = self.params[PROTO_KEY]
                        p.copy_(F.normalize(p, dim=-1, eps=1e-6))

                total += loss_per.mean().item()
                nb += 1
        return total / max(nb, 1)

    @torch.no_grad()
    def evaluate(self, batch_size):
        """Per-client masked-MAE val loss → (M,) on CPU."""
        self.base.eval()
        vfwd = vmap(self._fwd)
        X, Y, Tids = self.Xval, self.Yval, self.Tval
        N = X.shape[1]
        se = torch.zeros(self.M, device=self.device)
        cnt = torch.zeros(self.M, device=self.device)
        for b in range(0, N, batch_size):
            xb, yb = X[:, b:b + batch_size], Y[:, b:b + batch_size]
            tb = Tids[:, b:b + batch_size] if Tids is not None else None
            y_hat = vfwd(self.params, self.buffers, xb, tb)
            mask = (yb * self.std + self.mean > self.mask_threshold).float()
            se += ((y_hat - yb).abs() * mask).flatten(1).sum(1)
            cnt += mask.flatten(1).sum(1)
        self.base.train()
        return (se / cnt.clamp(min=1.0)).cpu()

    @torch.no_grad()
    def evaluate_raw_mae(self, batch_size):
        """Per-client raw-scale validation MAE records for worst-client logs."""
        self.base.eval()
        vfwd = vmap(self._fwd)
        X, Y, Tids = self.Xval, self.Yval, self.Tval
        N = X.shape[1]
        se = torch.zeros(self.M, device=self.device)
        cnt = torch.zeros(self.M, device=self.device)
        for b in range(0, N, batch_size):
            xb, yb = X[:, b:b + batch_size], Y[:, b:b + batch_size]
            tb = Tids[:, b:b + batch_size] if Tids is not None else None
            y_hat = vfwd(self.params, self.buffers, xb, tb)
            y_true = yb * self.std + self.mean
            y_pred = y_hat * self.std + self.mean
            mask = y_true > self.mask_threshold
            se += (torch.abs(y_pred - y_true) * mask).flatten(1).sum(1)
            cnt += mask.flatten(1).sum(1)
        self.base.train()

        mae = (se / cnt.clamp(min=1.0)).cpu().numpy()
        std = self.std.view(self.M).detach().cpu().numpy()
        return [
            {
                "client_id": int(c.client_id),
                "raw_mae": float(mae[m]),
                "target_std": float(std[m]),
            }
            for m, c in enumerate(self.clients)
        ]

    def sync_to_clients(self):
        """Write stacked params back into each Client.model (for snapshot/eval)."""
        names = list(self.params.keys())
        for m, client in enumerate(self.clients):
            sd = {k: self.params[k][m].detach() for k in names}
            client.model.load_state_dict(sd, strict=True)
