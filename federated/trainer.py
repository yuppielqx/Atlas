"""Federated training loop with optional multi-GPU parallel client training."""

import os
import csv
import copy
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .client import Client
from .server import Server
from models.causal_fedst import CausalFedST


def build_clients_and_server(
    clients_data,
    model_kwargs,
    batch_size=32,
    lr=1e-3,
    local_epochs=5,
    device="cpu",
    weight_decay=1e-4,
    align_mode="none",
    align_eps=0.05,
    align_iters=50,
    mi_weight=0.0,
    freeze_prototypes=False,
    initial_prototypes=None,
    inv_weight=0.0,
    freeze_encoder=False,
    fedprox_mu=0.0,
    ditto_lambda=0.1,
    normalize_codebook=True,
    codebook_noise_std=0.0,
    codebook_noise_type="gaussian",
):
    """Build client and server instances from data and model config.

    Args:
        clients_data: list of per-client dataset dicts from data.dataset
        model_kwargs: dict of CausalFedST constructor arguments
        batch_size: local batch size
        lr: learning rate
        local_epochs: local epochs per round
        device: torch device
        align_mode: "none" | "ot" | "hungarian" — pre-aggregation prototype alignment
        align_eps: Sinkhorn entropic regularization (ot only)
        align_iters: Sinkhorn iteration count (ot only)

    Returns:
        clients: list of Client instances
        server: Server instance
    """
    n_clients = len(clients_data)
    hid_dim = model_kwargs.get("hid_dim", 64)
    K = model_kwargs.get("K", 10)

    # Create server
    server = Server(K=K, hid_dim=hid_dim, device=device,
                    align_mode=align_mode, align_eps=align_eps, align_iters=align_iters,
                    normalize_codebook=normalize_codebook)

    # Initial prototypes on the unit hypersphere (consistent with the server's
    # spherical-FedAvg constraint). For transfer experiments a pretrained
    # codebook can be supplied via ``initial_prototypes``; otherwise a fresh
    # template's random prototypes are used.
    import torch.nn.functional as F
    if initial_prototypes is None:
        template = CausalFedST(**model_kwargs)
        initial_prototypes = template.get_prototypes()
    if normalize_codebook:
        initial_prototypes = F.normalize(initial_prototypes, dim=-1, eps=1e-6)
    server.global_prototypes = initial_prototypes.to(device)

    # Create clients
    clients = []
    for i, data in enumerate(clients_data):
        model = CausalFedST(**model_kwargs)
        model.set_prototypes(initial_prototypes.clone())

        client = Client(
            client_id=int(data.get("client_id", i)),
            model=model,
            X_train=data["X_train"],
            Y_train=data["Y_train"],
            T_train=data.get("T_train"),
            X_val=data.get("X_val"),
            Y_val=data.get("Y_val"),
            T_val=data.get("T_val"),
            batch_size=batch_size,
            lr=lr,
            local_epochs=local_epochs,
            device=device,
            weight_decay=weight_decay,
            target_mean=data.get("target_mean"),
            target_std=data.get("target_std"),
            mi_weight=mi_weight,
            freeze_prototypes=freeze_prototypes,
            inv_weight=inv_weight,
            freeze_encoder=freeze_encoder,
        )
        # FedProx strength rides along as a client attribute so the vectorized
        # path can read it without changing the Client constructor.
        client.fedprox_mu = fedprox_mu
        client.ditto_lambda = ditto_lambda
        client.codebook_noise_std = codebook_noise_std
        client.codebook_noise_type = codebook_noise_type
        clients.append(client)

    if ditto_lambda > 0:
        for client in clients:
            client.enable_ditto(client.model, ditto_lambda)

    return clients, server


def _train_gpu_group(clients, indices, fl_mode="causal_proto"):
    """Train a group of clients (already on their assigned GPU). Called from a thread."""
    results = []
    for idx in indices:
        client = clients[idx]
        if fl_mode == "ditto":
            loss, protos = client.train_ditto()
        else:
            loss, protos = client.train_local()
        results.append((idx, loss, protos.cpu().clone(), len(client.train_loader.dataset)))
    return results


def _eval_gpu_group(clients, indices, loader="val"):
    """Evaluate a group of clients on their assigned GPU. Called from a thread."""
    results = []
    for idx in indices:
        vl = clients[idx].evaluate(loader)
        results.append((idx, vl))
    return results


def _eval_raw_gpu_group(clients, indices, loader="val"):
    """Raw-scale MAE records for a group of clients."""
    return [clients[idx].evaluate_raw_mae(loader) for idx in indices]


# Which state_dict keys each fedavg* mode shares across clients.
_FEDAVG_KEY_PRED = {
    "fedavg": lambda k: not k.startswith("prototype."),     # encoder + predictor
    "fedavg_proto": lambda k: True,                          # everything incl. prototype
    "fedavg_enc": lambda k: k.startswith("encoder."),        # encoder only (no proto)
    "fedavg_enc_proto": lambda k: k.startswith("encoder."),  # encoder only (proto local)
}
_CAUSAL_PROTO_MODES = {"causal_proto", "causal_proto_enc"}


def _fedavg_state_dicts(src_clients, weights, key_pred, dst_clients=None):
    """In-place weighted FedAvg of selected state_dict entries.

    Averages the keys matching ``key_pred`` over ``src_clients`` (weighted by
    ``weights``) and loads the result into every ``dst_clients`` model.
    """
    dst_clients = dst_clients if dst_clients is not None else src_clients
    sds = [c.model.state_dict() for c in src_clients]
    keys = [k for k in sds[0] if key_pred(k)]
    avg = {}
    for k in keys:
        # Accumulate on CPU so clients pinned to different GPUs can be averaged;
        # load_state_dict copies the result back onto each client's device/dtype.
        acc = sds[0][k].detach().float().cpu() * weights[0]
        for sd, w in zip(sds[1:], weights[1:]):
            acc = acc + sd[k].detach().float().cpu() * w
        avg[k] = acc.to(sds[0][k].dtype)
    for c in dst_clients:
        c.model.load_state_dict(avg, strict=False)


def _fedavg_global_models(src_clients, weights, dst_clients=None):
    """Weighted FedAvg over Ditto global models, then broadcast to all clients."""
    dst_clients = dst_clients if dst_clients is not None else src_clients
    sds = [c.global_model.state_dict() for c in src_clients]
    keys = list(sds[0].keys())
    avg = {}
    for k in keys:
        if not torch.is_floating_point(sds[0][k]):
            avg[k] = sds[0][k].detach().cpu().clone()
            continue
        acc = sds[0][k].detach().float().cpu() * weights[0]
        for sd, w in zip(sds[1:], weights[1:]):
            acc = acc + sd[k].detach().float().cpu() * w
        avg[k] = acc.to(sds[0][k].dtype)
    for c in dst_clients:
        c.global_model.load_state_dict(avg, strict=False)
        c.global_model.to(c.device)
        c.global_optimizer = torch.optim.Adam(
            [p for p in c.global_model.parameters() if p.requires_grad],
            lr=c.lr,
            weight_decay=c.weight_decay,
        )


def _snapshot_state(clients, server):
    """Capture a CPU copy of all client models + global prototypes."""
    return {
        "client_models": [
            {k: v.detach().cpu().clone() for k, v in c.model.state_dict().items()}
            for c in clients
        ],
        "prototypes": server.get_prototypes().detach().cpu().clone(),
    }


def _restore_state(clients, server, snapshot):
    """Load a snapshot from _snapshot_state back into clients and server."""
    for c, sd in zip(clients, snapshot["client_models"]):
        c.model.load_state_dict(sd)
    server.global_prototypes = snapshot["prototypes"].to(server.device)


def _get_lr(optimizers):
    """Read the first optimizer LR for logging/scheduler state."""
    if not optimizers:
        return float("nan")
    return float(optimizers[0].param_groups[0]["lr"])


def _set_lr(optimizers, lr):
    """Set LR on every param group of every optimizer."""
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = lr


def _maybe_reduce_lr(
    optimizers, improved, lr_bad_rounds, lr_scheduler, lr_factor,
    lr_patience, min_lr, verbose, rnd,
):
    """Plateau scheduler stepped once per federated round after validation."""
    if lr_scheduler == "none" or lr_patience is None or lr_patience <= 0:
        return 0 if improved else lr_bad_rounds, False
    if lr_scheduler != "plateau":
        raise ValueError(f"Unsupported lr_scheduler={lr_scheduler!r}")
    if not 0.0 < lr_factor < 1.0:
        raise ValueError("lr_factor must be in (0, 1) for plateau scheduler")

    if improved:
        return 0, False

    lr_bad_rounds += 1
    if lr_bad_rounds < lr_patience:
        return lr_bad_rounds, False

    old_lr = _get_lr(optimizers)
    new_lr = max(old_lr * lr_factor, min_lr)
    if new_lr < old_lr - 1e-12:
        _set_lr(optimizers, new_lr)
        if verbose:
            print(f"    lr[plateau]: round {rnd} reducing {old_lr:.3e} -> {new_lr:.3e}")
        return 0, True
    return 0, False


# ── per-round logging helpers ──
_CSV_HEADER = ["round", "train_loss", "val_mean", "val_min", "val_median",
               "val_max", "best_val", "best_round", "no_improve", "proto_drift_rel"]
_CLIENT_VAL_CSV_HEADER = ["round", "client_id", "val_loss", "raw_mae", "target_std"]


def _fmt_float(value):
    value = float(value)
    return f"{value:.6f}" if np.isfinite(value) else str(value)


def _client_val_csv_path(metrics_csv_path):
    if not metrics_csv_path:
        return None
    return os.path.join(os.path.dirname(metrics_csv_path), "client_val_loss.csv")


def _record_client_val_losses(history, rnd, client_ids, val_arr, raw_records, metrics_csv_path):
    """Store and optionally write per-client validation losses for this round."""
    client_ids = [int(c) for c in client_ids]
    vals = [float(v) for v in np.asarray(val_arr, dtype=float).reshape(-1)]
    if "client_ids" not in history:
        history["client_ids"] = client_ids
    history.setdefault("client_val_loss", []).append(vals)

    csv_path = _client_val_csv_path(metrics_csv_path)
    if not csv_path:
        return

    raw_by_client = {int(r.get("client_id")): r for r in raw_records or []}
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(_CLIENT_VAL_CSV_HEADER)
        for client_id, val_loss in zip(client_ids, vals):
            raw = raw_by_client.get(client_id, {})
            w.writerow([
                rnd,
                client_id,
                _fmt_float(val_loss),
                _fmt_float(raw.get("raw_mae", float("nan"))),
                _fmt_float(raw.get("target_std", float("nan"))),
            ])


def _append_nan_client_val_losses(history, clients):
    client_ids = [int(c.client_id) for c in clients]
    if "client_ids" not in history:
        history["client_ids"] = client_ids
    history.setdefault("client_val_loss", []).append([float("nan")] * len(client_ids))


def _proto_drift(prev, new):
    """Relative L2 change of the global prototypes: ‖P_new − P_prev‖ / ‖P_prev‖."""
    return (new - prev).norm().item() / (prev.norm().item() + 1e-8)


def _log_raw_worst(records, prefix="val", top_k=5, verbose=True):
    """Print top-k worst clients by raw-scale MAE."""
    if not verbose or not records:
        return []
    clean = [r for r in records if np.isfinite(r.get("raw_mae", float("nan")))]
    clean.sort(key=lambda r: r["raw_mae"], reverse=True)
    worst = clean[:top_k]
    if worst:
        msg = ", ".join(
            f"id={r['client_id']} mae={r['raw_mae']:.2f} std={r['target_std']:.2f}"
            for r in worst
        )
        print(f"    {prefix} raw worst-{len(worst)}: {msg}")
    return worst


def _all_finite_prototypes(prototypes):
    return all(torch.isfinite(p).all().item() for p in prototypes)


def _log_align(server, verbose):
    """Print one line of prototype-alignment diagnostics, if alignment is on."""
    if not verbose or server.align_mode == "none" or not server.last_align_diag:
        return
    d = server.last_align_diag
    tqdm.write(
        f"    align[{server.align_mode}]: "
        f"perm_frac={d['perm_frac']:.3f} mean_shift={d['mean_shift']:.3f}"
    )


def _log_round(rnd, train_loss, val_arr, best_val, best_round, no_improve,
               patience, proto_drift, csv_path, verbose):
    """Print a detailed per-round line and append a row to metrics.csv.

    Returns (mean, min, median, max) of the per-client validation losses.
    """
    v = np.asarray(val_arr, dtype=float)
    mean, vmin, vmed, vmax = float(v.mean()), float(v.min()), float(np.median(v)), float(v.max())

    if verbose:
        pat = f"/{patience}" if patience else ""
        tqdm.write(
            f"[R{rnd:>3}] train {train_loss:.4f} | "
            f"val {mean:.4f} (min {vmin:.3f} / med {vmed:.3f} / max {vmax:.3f}) | "
            f"best {best_val:.4f}@{best_round} (no_imp {no_improve}{pat}) | "
            f"ΔP {proto_drift:.2e}"
        )

    if csv_path:
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(_CSV_HEADER)
            w.writerow([rnd, f"{train_loss:.6f}", f"{mean:.6f}", f"{vmin:.6f}",
                        f"{vmed:.6f}", f"{vmax:.6f}", f"{best_val:.6f}", best_round,
                        no_improve, f"{proto_drift:.6e}"])

    return mean, vmin, vmed, vmax


def federated_train(
    clients,
    server,
    n_rounds=50,
    fraction=1.0,
    n_gpus=1,
    device="cpu",
    verbose=True,
    patience=None,
    min_delta=0.0,
    vectorized=False,
    fl_mode="causal_proto",
    metrics_csv_path=None,
    lr_scheduler="plateau",
    lr_factor=0.5,
    lr_patience=3,
    min_lr=1e-6,
    freeze_prototypes=False,
    freeze_encoder=False,
    sparse_proto_upload=False,
    proto_topk=1,
    sparse_warmup_rounds=0,
    normalize_codebook=True,
    codebook_noise_std=0.0,
    codebook_noise_type="gaussian",
):
    """Run federated training.

    When n_gpus > 1, both client training and per-round validation are
    parallelized across GPUs using threads (CUDA operations release the GIL, so
    threads on different GPUs run concurrently). Clients are pinned to GPUs
    round-robin at the start.

    When ``vectorized`` is True, the clients pinned to each GPU are processed as
    a single vmapped ensemble (one batched kernel per step) instead of a Python
    loop, removing the within-GPU serialization. Requires aligned client shapes
    (equal sample counts, true for METR-LA) and trains all clients every round
    (fraction is ignored). See federated/vectorized.py.

    Validation loss is tracked every round; the state (all client models +
    global prototypes) at the best validation round is snapshotted and restored
    before returning, so the final/evaluated model is the best one — not the
    last (potentially overfit) one. If ``patience`` is set, training stops early
    after that many rounds without improvement.

    Args:
        clients: list of Client instances
        server: Server instance
        n_rounds: number of communication rounds
        fraction: fraction of clients to sample per round (1.0 = all)
        n_gpus: number of GPUs for parallel client training (1 = sequential)
        device: torch device (used for server and single-GPU mode)
        verbose: print progress
        patience: stop after this many rounds with no val improvement
                  (None = no early stopping, run all rounds)
        min_delta: minimum decrease in val loss to count as an improvement
        lr_scheduler: "none" or "plateau"; plateau lowers LR on stalled val loss
        lr_factor: multiplicative LR decay factor for plateau scheduler
        lr_patience: rounds without val improvement before LR decay
        min_lr: lower bound for adaptive LR

    Returns:
        history: dict with keys (train_loss, val_loss) each a list of per-round
                 values, plus "best_round" (0-indexed) and "best_val_loss".
    """
    if vectorized:
        return _federated_train_vectorized(
            clients, server, n_rounds=n_rounds, n_gpus=n_gpus, device=device,
            verbose=verbose, patience=patience, min_delta=min_delta,
            fl_mode=fl_mode,
            metrics_csv_path=metrics_csv_path, lr_scheduler=lr_scheduler,
            lr_factor=lr_factor, lr_patience=lr_patience, min_lr=min_lr,
            freeze_prototypes=freeze_prototypes,
            freeze_encoder=freeze_encoder,
            normalize_codebook=normalize_codebook,
            codebook_noise_std=codebook_noise_std,
            codebook_noise_type=codebook_noise_type,
        )

    n_clients = len(clients)
    n_selected = max(1, int(n_clients * fraction))
    history = {"train_loss": [], "val_loss": [],
               "val_min": [], "val_median": [], "val_max": [], "proto_drift": [],
               "client_val_loss": [], "client_ids": [int(c.client_id) for c in clients],
               "lr": []}

    best_val = float("inf")
    best_round = -1
    best_snapshot = None
    rounds_no_improve = 0
    lr_bad_rounds = 0

    # Sparse upload: the per-client path with variable-length returns needs a
    # single-GPU loop.  n_gpus>1 would require a separate _train_gpu_group_sparse;
    # for simplicity we serialise when sparse upload is active.
    if sparse_proto_upload and n_gpus > 1 and torch.cuda.is_available():
        if verbose:
            print("  [warn] --sparse_proto_upload forces n_gpus=1 "
                  "(multi-GPU sparse dispatch not yet implemented)")
        n_gpus = 1

    # Assign clients to GPUs round-robin (only for multi-GPU)
    if n_gpus > 1 and torch.cuda.is_available():
        n_gpus = min(n_gpus, torch.cuda.device_count())
        for i, client in enumerate(clients):
            gpu_id = i % n_gpus
            client.model.to(f"cuda:{gpu_id}")
            client.device = f"cuda:{gpu_id}"
        print(f"  Clients distributed across {n_gpus} GPUs "
              f"({', '.join(f'GPU {g}: {sum(1 for j in range(n_clients) if j % n_gpus == g)} clients' for g in range(n_gpus))})")
    elif n_gpus > 1 and not torch.cuda.is_available():
        print("  Warning: n_gpus > 1 but CUDA not available, falling back to single device")
        n_gpus = 1
        for client in clients:
            client.model.to(device)
            client.device = device
    else:
        for client in clients:
            client.model.to(device)
            client.device = device

    # Per-round prototype snapshots for post-hoc analysis (t-SNE trajectories).
    PROTO_SAVE_EVERY = 10

    pbar = tqdm(range(n_rounds), desc="Federated Training", disable=not verbose)
    for rnd in pbar:
        # Select clients
        selected = np.random.choice(n_clients, n_selected, replace=False)

        # Broadcast global prototypes to ALL clients (only when the federated
        # object is the prototype set; fedavg*/local modes manage params directly).
        if fl_mode in _CAUSAL_PROTO_MODES:
            global_protos = server.broadcast()
            for client in clients:
                client.set_prototypes(global_protos)

        # ── Local training ──
        trained_indices = []
        # Sparse prototype upload: after warmup rounds, clients only send the
        # top-k dominant prototype vectors + their indices.
        sparsity_active = sparse_proto_upload and rnd >= sparse_warmup_rounds
        if n_gpus <= 1:
            # Sequential
            client_protos = []
            sparse_updates = []
            train_losses = []
            dataset_sizes = []
            for idx in selected:
                if fl_mode == "ditto":
                    loss, protos = clients[idx].train_ditto()
                    trained_indices.append(idx)
                    client_protos.append(protos)
                elif sparsity_active:
                    loss, indices, vecs = clients[idx].train_local_sparse(
                        proto_topk=proto_topk)
                    trained_indices.append(idx)
                    sparse_updates.append((indices, vecs))
                else:
                    loss, protos = clients[idx].train_local()
                    trained_indices.append(idx)
                    client_protos.append(protos)
                train_losses.append(loss)
                dataset_sizes.append(len(clients[idx].train_loader.dataset))
        else:
            # Parallel across GPUs: group selected clients by their assigned GPU
            gpu_groups = {g: [] for g in range(n_gpus)}
            for idx in selected:
                gpu_groups[idx % n_gpus].append(idx)

            client_protos = []
            train_losses = []
            dataset_sizes = []

            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = []
                for gpu_id, group_indices in gpu_groups.items():
                    if group_indices:
                        futures.append(executor.submit(
                            _train_gpu_group, clients, group_indices, fl_mode))

                for future in as_completed(futures):
                    for idx, loss, protos, n_train in future.result():
                        trained_indices.append(idx)
                        client_protos.append(protos)
                        train_losses.append(loss)
                        dataset_sizes.append(n_train)

        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        if not np.isfinite(train_mean):
            if verbose:
                print(f"\n  Non-finite local loss at round {rnd}; "
                      "stopping before prototype aggregation.")
            history["train_loss"].append(train_mean)
            history["val_loss"].append(float("nan"))
            history["val_min"].append(float("nan"))
            history["val_median"].append(float("nan"))
            history["val_max"].append(float("nan"))
            history["proto_drift"].append(float("nan"))
            _append_nan_client_val_losses(history, clients)
            break
        elif not sparsity_active and not _all_finite_prototypes(client_protos):
            if verbose:
                print(f"\n  Non-finite prototypes at round {rnd}; "
                      "stopping before prototype aggregation.")
            history["train_loss"].append(train_mean)
            history["val_loss"].append(float("nan"))
            history["val_min"].append(float("nan"))
            history["val_median"].append(float("nan"))
            history["val_max"].append(float("nan"))
            history["proto_drift"].append(float("nan"))
            _append_nan_client_val_losses(history, clients)
            break

        # ── Server aggregation (dispatched by fl_mode) ──
        total_size = sum(dataset_sizes)
        weights = [s / total_size for s in dataset_sizes]
        if fl_mode in _CAUSAL_PROTO_MODES:
            drift = 0.0
            if not freeze_prototypes:
                prev_protos = server.get_prototypes().detach().cpu().clone()
                if sparsity_active:
                    server.aggregate_sparse(sparse_updates, weights)
                else:
                    server.aggregate(client_protos, weights)
                drift = _proto_drift(prev_protos, server.get_prototypes().detach().cpu())
                # Validate/checkpoint the deployable state: personalized client weights
                # paired with the same aggregated global prototypes.
                global_protos = server.broadcast()
                for client in clients:
                    client.set_prototypes(global_protos)
                # Save prototype snapshot every PROTO_SAVE_EVERY rounds (and round 0).
                if rnd % PROTO_SAVE_EVERY == 0:
                    history.setdefault("proto_history", []).append(
                        server.get_prototypes().detach().cpu().clone())
                    history.setdefault("proto_history_rounds", []).append(rnd)
            if fl_mode == "causal_proto_enc" and not freeze_encoder:
                _fedavg_state_dicts([clients[i] for i in trained_indices], weights,
                                    _FEDAVG_KEY_PRED["fedavg_enc"], dst_clients=clients)
        elif fl_mode in _FEDAVG_KEY_PRED:
            if freeze_prototypes:
                # Preserve transfer-mode semantics for legacy fedavg* runs:
                # fixed loaded prototypes mean no cross-client sync this round.
                drift = 0.0
            else:
                # Weighted FedAvg of selected clients' params → broadcast to all clients.
                _fedavg_state_dicts([clients[i] for i in trained_indices], weights,
                                    _FEDAVG_KEY_PRED[fl_mode], dst_clients=clients)
                drift = 0.0
        elif fl_mode == "ditto":
            _fedavg_global_models([clients[i] for i in trained_indices], weights,
                                  dst_clients=clients)
            drift = 0.0
        else:  # local, local_proto: no aggregation, fully personalized
            drift = 0.0

        history["train_loss"].append(train_mean)

        # ── Validation ── (parallelized across GPUs, like training)
        val_by_client = np.full(n_clients, np.nan, dtype=float)
        if n_gpus <= 1:
            for idx, client in enumerate(clients):
                vl = client.evaluate("val")
                val_by_client[idx] = vl
        else:
            eval_groups = {g: [] for g in range(n_gpus)}
            for idx in range(n_clients):
                eval_groups[idx % n_gpus].append(idx)
            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = [
                    executor.submit(_eval_gpu_group, clients, group, "val")
                    for group in eval_groups.values() if group
                ]
                for future in as_completed(futures):
                    for idx, vl in future.result():
                        val_by_client[idx] = vl
        finite_val = val_by_client[np.isfinite(val_by_client)]
        val_losses = finite_val.tolist()
        cur_val = float(finite_val.mean()) if finite_val.size else float("nan")
        history["val_loss"].append(cur_val)

        raw_records = []
        if n_gpus <= 1:
            raw_records = [c.evaluate_raw_mae("val") for c in clients]
        else:
            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = [
                    executor.submit(_eval_raw_gpu_group, clients, group, "val")
                    for group in eval_groups.values() if group
                ]
                for future in as_completed(futures):
                    raw_records.extend(future.result())

        # ── Track best model / early stopping ──
        improved = cur_val == cur_val and cur_val < best_val - min_delta  # NaN-safe
        if improved:
            best_val = cur_val
            best_round = rnd
            best_snapshot = _snapshot_state(clients, server)
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
        lr_bad_rounds, _ = _maybe_reduce_lr(
            [c.optimizer for c in clients], improved, lr_bad_rounds,
            lr_scheduler, lr_factor, lr_patience, min_lr, verbose, rnd,
        )
        history["lr"].append(_get_lr([clients[0].optimizer]))

        # ── Detailed per-round log + metrics.csv ──
        _, vmin, vmed, vmax = _log_round(
            rnd, history["train_loss"][-1], finite_val if finite_val.size else [float("nan")],
            best_val, best_round, rounds_no_improve, patience, drift,
            metrics_csv_path, verbose)
        _record_client_val_losses(
            history, rnd, [c.client_id for c in clients], val_by_client,
            raw_records, metrics_csv_path)
        _log_raw_worst(raw_records, prefix="val", top_k=5, verbose=verbose)
        _log_align(server, verbose)
        history["val_min"].append(vmin)
        history["val_median"].append(vmed)
        history["val_max"].append(vmax)
        history["proto_drift"].append(drift)

        if patience is not None and rounds_no_improve >= patience:
            if verbose:
                print(f"\n  Early stopping at round {rnd} "
                      f"(no val improvement for {patience} rounds; "
                      f"best={best_val:.4f} @ round {best_round})")
            break

    # Restore the best-validation state so evaluation uses it, not the last round
    if best_snapshot is not None:
        _restore_state(clients, server, best_snapshot)
        if verbose:
            print(f"  Restored best model: val={best_val:.4f} @ round {best_round}")

    history["best_round"] = best_round
    history["best_val_loss"] = best_val

    # Move models back to main device for final evaluation
    if n_gpus > 1 and torch.cuda.is_available():
        for client in clients:
            client.model.to(device)
            client.device = device

    return history


def _federated_train_vectorized(
    clients, server, n_rounds, n_gpus, device, verbose, patience, min_delta,
    fl_mode="causal_proto",
    metrics_csv_path=None, lr_scheduler="plateau", lr_factor=0.5,
    lr_patience=3, min_lr=1e-6, freeze_prototypes=False, freeze_encoder=False,
    normalize_codebook=True, codebook_noise_std=0.0,
    codebook_noise_type="gaussian",
):
    """Federated training where each GPU's clients run as one vmapped ensemble.

    Mirrors federated_train's round structure (broadcast → local train → FedAvg
    → validate → best-model early stopping) but replaces the per-client Python
    loop with a batched ensemble per GPU. All clients train every round.
    """
    from .vectorized import VectorGroup

    n_clients = len(clients)

    # Hyperparameters are read back from the already-built clients so this path
    # needs no extra plumbing.
    c0 = clients[0]
    local_epochs = c0.local_epochs
    batch_size = c0.train_loader.batch_size
    lr = c0.optimizer.param_groups[0]["lr"]
    weight_decay = c0.optimizer.param_groups[0]["weight_decay"]
    mask_threshold = getattr(c0, "mask_threshold", 0.1)

    # Partition clients across GPUs (round-robin), one VectorGroup per device.
    if n_gpus > 1 and torch.cuda.is_available():
        n_gpus = min(n_gpus, torch.cuda.device_count())
        group_devices = [f"cuda:{g}" for g in range(n_gpus)]
    else:
        n_gpus = 1
        group_devices = [str(device)]

    groups, group_idx = [], []
    for g, gdev in enumerate(group_devices):
        idxs = [i for i in range(n_clients) if i % n_gpus == g]
        if not idxs:
            continue
        groups.append(VectorGroup(
            [clients[i] for i in idxs], device=gdev, lr=lr,
            weight_decay=weight_decay, mask_threshold=mask_threshold,
            mi_weight=getattr(clients[0], "mi_weight", 0.0),
            freeze_prototypes=freeze_prototypes,
            inv_weight=getattr(clients[0], "inv_weight", 0.0),
            freeze_encoder=freeze_encoder,
            fedprox_mu=getattr(clients[0], "fedprox_mu", 0.0),
            normalize_codebook=normalize_codebook,
            codebook_noise_std=codebook_noise_std,
            codebook_noise_type=codebook_noise_type,
        ))
        group_idx.append(idxs)
    if verbose:
        print(f"  [vectorized] {len(groups)} GPU ensemble(s): "
              f"{', '.join(f'{gd}:{g.M} clients' for gd, g in zip(group_devices, groups))}")

    history = {"train_loss": [], "val_loss": [],
               "val_min": [], "val_median": [], "val_max": [], "proto_drift": [],
               "client_val_loss": [], "client_ids": [int(c.client_id) for c in clients],
               "lr": []}
    best_val, best_round, best_snapshot, rounds_no_improve = float("inf"), -1, None, 0
    lr_bad_rounds = 0

    PROTO_SAVE_EVERY = 10

    pbar = tqdm(range(n_rounds), desc="Federated Training (vec)", disable=not verbose)
    for rnd in pbar:
        if fl_mode in _CAUSAL_PROTO_MODES:
            global_protos = server.broadcast()
            for grp in groups:
                grp.set_global_prototype(global_protos)

        # ── Local training (parallel across GPUs) ──
        if len(groups) == 1:
            group_losses = [groups[0].train_local_epochs(local_epochs, batch_size)]
        else:
            with ThreadPoolExecutor(max_workers=len(groups)) as ex:
                futs = {ex.submit(g.train_local_epochs, local_epochs, batch_size): gi
                        for gi, g in enumerate(groups)}
                group_losses = [None] * len(groups)
                for fut in as_completed(futs):
                    group_losses[futs[fut]] = fut.result()

        train_mean = float(np.average(group_losses, weights=[g.M for g in groups]))
        if not np.isfinite(train_mean):
            if verbose:
                print(f"\n  Non-finite local loss at round {rnd}; "
                      "stopping before prototype aggregation/sync.")
            history["train_loss"].append(train_mean)
            history["val_loss"].append(float("nan"))
            history["val_min"].append(float("nan"))
            history["val_median"].append(float("nan"))
            history["val_max"].append(float("nan"))
            history["proto_drift"].append(float("nan"))
            _append_nan_client_val_losses(history, clients)
            break

        # ── Server aggregation (dispatched by fl_mode) ──
        if fl_mode in _CAUSAL_PROTO_MODES:
            drift = 0.0
            if not freeze_prototypes:
                prev_protos = server.get_prototypes().detach().cpu().clone()
                all_protos, all_sizes = [], []
                for grp in groups:
                    protos = grp.get_client_prototypes()        # (M, K, D)
                    sizes = grp.client_sizes()
                    all_protos.extend(protos[m] for m in range(grp.M))
                    all_sizes.extend(sizes)
                if not _all_finite_prototypes(all_protos):
                    if verbose:
                        print(f"\n  Non-finite prototypes at round {rnd}; "
                              "stopping before prototype aggregation/sync.")
                    history["train_loss"].append(train_mean)
                    history["val_loss"].append(float("nan"))
                    history["val_min"].append(float("nan"))
                    history["val_median"].append(float("nan"))
                    history["val_max"].append(float("nan"))
                    history["proto_drift"].append(float("nan"))
                    _append_nan_client_val_losses(history, clients)
                    break
                total = float(sum(all_sizes))
                server.aggregate(all_protos, [s / total for s in all_sizes])
                drift = _proto_drift(prev_protos, server.get_prototypes().detach().cpu())
                global_protos = server.broadcast()
                for grp in groups:
                    grp.set_global_prototype(global_protos)
                # Save prototype snapshot every PROTO_SAVE_EVERY rounds (and round 0).
                if rnd % PROTO_SAVE_EVERY == 0:
                    history.setdefault("proto_history", []).append(
                        server.get_prototypes().detach().cpu().clone())
                    history.setdefault("proto_history_rounds", []).append(rnd)
            if fl_mode == "causal_proto_enc" and not freeze_encoder:
                if len(groups) != 1:
                    raise RuntimeError("causal_proto_enc vectorized path requires one group")
                groups[0].fedavg_param_keys(_FEDAVG_KEY_PRED["fedavg_enc"])
        elif fl_mode in _FEDAVG_KEY_PRED:
            drift = 0.0
            if not freeze_prototypes:
                # Single-group param-FedAvg (main.py forces this path to one group):
                # average matching keys across the client dim and broadcast in place.
                groups[0].fedavg_param_keys(_FEDAVG_KEY_PRED[fl_mode])
        else:  # local, local_proto: no aggregation
            drift = 0.0

        # Sync stacked params back into Client.model so snapshot / final eval work.
        for grp in groups:
            grp.sync_to_clients()

        # Train loss = sample-weighted mean of per-group means (groups equal here).
        history["train_loss"].append(train_mean)

        # ── Validation (parallel across GPUs) ──
        if len(groups) == 1:
            val_per = [groups[0].evaluate(batch_size)]
        else:
            with ThreadPoolExecutor(max_workers=len(groups)) as ex:
                futs = {ex.submit(g.evaluate, batch_size): gi for gi, g in enumerate(groups)}
                val_per = [None] * len(groups)
                for fut in as_completed(futs):
                    val_per[futs[fut]] = fut.result()
        val_by_client = np.full(n_clients, np.nan, dtype=float)
        for idxs, vals in zip(group_idx, val_per):
            group_vals = vals.numpy()
            for local_i, client_idx in enumerate(idxs):
                val_by_client[client_idx] = group_vals[local_i]
        finite_val = val_by_client[np.isfinite(val_by_client)]
        cur_val = float(finite_val.mean()) if finite_val.size else float("nan")
        history["val_loss"].append(cur_val)

        if len(groups) == 1:
            raw_records = groups[0].evaluate_raw_mae(batch_size)
        else:
            with ThreadPoolExecutor(max_workers=len(groups)) as ex:
                futs = {ex.submit(g.evaluate_raw_mae, batch_size): gi for gi, g in enumerate(groups)}
                raw_records = []
                for fut in as_completed(futs):
                    raw_records.extend(fut.result())

        # ── Best-model tracking / early stopping (on synced Client.model) ──
        improved = cur_val == cur_val and cur_val < best_val - min_delta  # NaN-safe
        if improved:
            best_val, best_round = cur_val, rnd
            best_snapshot = _snapshot_state(clients, server)
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
        lr_bad_rounds, _ = _maybe_reduce_lr(
            [g.opt for g in groups], improved, lr_bad_rounds,
            lr_scheduler, lr_factor, lr_patience, min_lr, verbose, rnd,
        )
        history["lr"].append(_get_lr([groups[0].opt]))

        # ── Detailed per-round log + metrics.csv ──
        _, vmin, vmed, vmax = _log_round(
            rnd, history["train_loss"][-1], finite_val if finite_val.size else [float("nan")],
            best_val, best_round, rounds_no_improve, patience, drift,
            metrics_csv_path, verbose)
        _record_client_val_losses(
            history, rnd, [c.client_id for c in clients], val_by_client,
            raw_records, metrics_csv_path)
        _log_raw_worst(raw_records, prefix="val", top_k=5, verbose=verbose)
        _log_align(server, verbose)
        history["val_min"].append(vmin)
        history["val_median"].append(vmed)
        history["val_max"].append(vmax)
        history["proto_drift"].append(drift)

        if patience is not None and rounds_no_improve >= patience:
            if verbose:
                print(f"\n  Early stopping at round {rnd} (no val improvement for "
                      f"{patience} rounds; best={best_val:.4f} @ round {best_round})")
            break

    if best_snapshot is not None:
        _restore_state(clients, server, best_snapshot)
        if verbose:
            print(f"  Restored best model: val={best_val:.4f} @ round {best_round}")

    history["best_round"] = best_round
    history["best_val_loss"] = best_val

    # Move all client models to the main eval device.
    for client in clients:
        client.model.to(device)
        client.device = device

    return history


def evaluate_all(clients, server, data_key="X_test", Y_key="Y_test", device="cpu"):
    """Evaluate all clients on their test data.

    Args:
        clients: list of Client instances
        server: Server instance
        data_key: key for test input data in client data dict
        Y_key: key for test target data
        device: torch device

    Returns:
        metrics: dict with keys (mae, rmse, mape) averaged across clients
    """
    from torch.utils.data import DataLoader, TensorDataset
    from utils.metrics import compute_all_metrics

    # Broadcast final prototypes
    global_protos = server.broadcast()
    for client in clients:
        client.set_prototypes(global_protos)

    all_y_true = []
    all_y_pred = []

    for client in clients:
        client.model.eval()
        client.model.to(device)

        # Access the raw data from client's data attribute
        # We need to find the test data – use the val_loader pattern
        test_loader = getattr(client, "test_loader", None)
        if test_loader is None:
            # Fallback: use val_loader as test
            test_loader = client.val_loader
        if test_loader is None:
            continue

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                y_hat, _, _ = client.model(batch_x)
                all_y_true.append(batch_y[..., :1].cpu().numpy())
                all_y_pred.append(y_hat.cpu().numpy())

    if not all_y_true:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    return compute_all_metrics(y_true, y_pred)
