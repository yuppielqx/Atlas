#!/usr/bin/env python3
"""E1: quantify whether KnowAir-PM25 Atlas slots track measured weather.

This is checkpoint-only analysis: no Atlas model parameters are updated.
Non-overlapping training windows fit a fixed weather probe; held-out test
windows supply every reported probe metric. The validation period has no
train-q90 high-precipitation points and cannot fit that probe. Window positions
are absolute integer indices into the two KnowAir memmaps, never reconstructed
from approximate timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True, help="server causal-FedTS root")
    p.add_argument("--data-dir", type=Path, required=True, help="directory containing KnowAir-PM25 and KnowAir-Full")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe-train-stride", type=int, default=12,
                   help="fixed training-window start stride for probe fitting")
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    p.add_argument("--permutation-reps", type=int, default=99)
    p.add_argument("--nonlinear-control", action="store_true",
                   help="also fit a histogram-boosted nonlinear PM25/calendar control")
    p.add_argument("--no-save-activations", action="store_true")
    return p.parse_args()


def load_memmaps(data_dir: Path):
    dirs = [data_dir / "KnowAir-PM25", data_dir / "KnowAir-Full"]
    descs = [json.loads((d / "desc.json").read_text()) for d in dirs]
    pm_shape, full_shape = [tuple(d["shape"]) for d in descs]
    if pm_shape[:2] != full_shape[:2] or pm_shape[2] != 3 or full_shape[2] != 18:
        raise ValueError(f"KnowAir array shape mismatch: {pm_shape} vs {full_shape}")
    pm = np.memmap(dirs[0] / "data.dat", dtype=np.float32, mode="r", shape=pm_shape)
    full = np.memmap(dirs[1] / "data.dat", dtype=np.float32, mode="r", shape=full_shape)
    if descs[0]["start_time_utc"] != descs[1]["start_time_utc"]:
        raise ValueError("KnowAir start_time_utc differs")
    pm_freq = descs[0]["frequency (minutes)"]
    full_freq = descs[1]["frequency_minutes"]
    if pm_freq != full_freq:
        raise ValueError("KnowAir frequencies differ")
    if not np.array_equal(pm[:, :, 0], full[:, :, 17]):
        raise ValueError("KnowAir-PM25 target is not identical to KnowAir-Full channel 17")
    ratios = descs[0]["regular_settings"]["TRAIN_VAL_TEST_RATIO"]
    if ratios != descs[1]["regular_settings"]["TRAIN_VAL_TEST_RATIO"]:
        raise ValueError("KnowAir split ratios differ")
    return pm, full, descs[0], int(pm_freq)


def split_bounds(length: int, ratios: list[float]) -> dict[str, tuple[int, int]]:
    train_end = int(length * ratios[0])
    val_end = train_end + int(length * ratios[1])
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, length)}


def exact_window_indices(start: int, end: int, window: int, horizon: int):
    """Mirror data/dataset.py split-then-sliding-window, without crossing splits."""
    n = end - start - window - horizon + 1
    if n <= 0:
        raise ValueError("split is too short for window + horizon")
    starts = start + np.arange(n, dtype=np.int32)
    last_input = starts + window - 1
    last_target = starts + window + horizon - 1
    assert starts[0] == start and last_target[-1] == end - 1
    return starts, last_input, last_target


def load_model(checkpoint: dict, repo: Path, device: torch.device):
    sys.path.insert(0, str(repo.resolve()))
    from models.causal_fedst import CausalFedST
    from models.encoders import build_encoder_kwargs

    cfg = checkpoint["config"]
    if cfg.get("dataset") != "knowair-pm25":
        raise ValueError("E1 requires a KnowAir-PM25 checkpoint")
    if cfg.get("graph_context", False):
        raise ValueError("E1 expects PM25-only inputs, not graph context")
    if cfg.get("fl_mode") != "causal_proto":
        raise ValueError("E1 primary analysis expects fl_mode=causal_proto")
    if cfg.get("use_residual", True):
        raise ValueError("E1 primary analysis expects use_residual=false")
    kw = dict(
        input_dim=1,
        hid_dim=cfg.get("hid_dim", 64),
        seq_len=cfg.get("window_size", 12),
        horizon=cfg.get("horizon", 12),
        K=cfg.get("K", 10),
        tau=cfg.get("tau", 0.1),
        stid_num_layers=cfg.get("stid_num_layers", 3),
        dropout=cfg.get("dropout", 0.1),
        use_residual=False,
        residual_mode="none",
        residual_scale=cfg.get("residual_scale", 1.0),
        residual_gate_init=cfg.get("residual_gate_init", -2.0),
        encoder_type=cfg.get("encoder_type", "stid"),
        encoder_kwargs=build_encoder_kwargs(cfg.get("encoder_type", "stid"), cfg.get("encoder_opt", [])),
        use_prototype=True,
        use_aux_inv=(cfg.get("aux_inv_weight", 0.0) > 0),
    )
    if "separate_residual" in inspect.signature(CausalFedST).parameters:
        kw["separate_residual"] = cfg.get("separate_residual", False)
    model = CausalFedST(**kw).to(device).eval()
    return model


def time_fields(desc: dict, minutes: int, absolute_index: np.ndarray, pm: np.ndarray):
    dt = pd.Timestamp(desc["start_time_utc"]).tz_convert("UTC").tz_localize(None)
    timestamps = pd.DatetimeIndex(dt + pd.to_timedelta(absolute_index.astype(np.int64) * minutes, unit="m"))
    tod = np.asarray(pm[absolute_index, 0, 1])
    dow = np.asarray(pm[absolute_index, 0, 2])
    expected_tod = (timestamps.hour * 60 + timestamps.minute) / 1440.0
    expected_dow = timestamps.dayofweek / 7.0
    if not np.allclose(tod, expected_tod, atol=1e-6):
        raise ValueError("PM25 time_of_day does not match absolute last-input index")
    if not np.allclose(dow, expected_dow, atol=1e-6):
        raise ValueError("PM25 day_of_week does not match absolute last-input index")
    return {
        "hour_slot": (timestamps.hour // 3).to_numpy(dtype=np.int16),
        "day_of_week": timestamps.dayofweek.to_numpy(dtype=np.int16),
        "month": timestamps.month.to_numpy(dtype=np.int16),
        "week": timestamps.to_period("W-SUN").start_time.strftime("%Y-%m-%d").to_numpy(),
    }


def extract_split(
    split: str, bounds: tuple[int, int], pm: np.ndarray, full: np.ndarray,
    desc: dict, minutes: int, checkpoint: dict, model: torch.nn.Module,
    device: torch.device, batch_size: int, stride: int = 1,
) -> dict[str, np.ndarray]:
    cfg = checkpoint["config"]
    window, horizon = int(cfg.get("window_size", 12)), int(cfg.get("horizon", 12))
    starts, last_input, last_target = exact_window_indices(*bounds, window, horizon)
    if stride < 1:
        raise ValueError("stride must be positive")
    starts, last_input, last_target = starts[::stride], last_input[::stride], last_target[::stride]
    fields = time_fields(desc, minutes, last_input, pm)
    n_windows = len(starts)
    meta = checkpoint.get("clients_data_meta")
    states = checkpoint["client_models"]
    if meta is None or len(meta) != len(states):
        raise ValueError("checkpoint must contain aligned clients_data_meta and client_models")
    k = int(cfg.get("K", 10))
    total = n_windows * len(states)
    result = {
        "weights": np.empty((total, k), dtype=np.float32),
        "client_id": np.empty(total, dtype=np.int16),
        "last_input_idx": np.empty(total, dtype=np.int32),
        "last_target_idx": np.empty(total, dtype=np.int32),
        "pm_window": np.empty((total, window), dtype=np.float32),
        "tp": np.empty(total, dtype=np.float32),
        "t2m": np.empty(total, dtype=np.float32),
        "humidity": np.empty(total, dtype=np.float32),
        "blh": np.empty(total, dtype=np.float32),
        "hour_slot": np.empty(total, dtype=np.int16),
        "day_of_week": np.empty(total, dtype=np.int16),
        "month": np.empty(total, dtype=np.int16),
        "week": np.empty(total, dtype="U10"),
    }
    q = checkpoint["server_prototypes"].float()
    offset = np.arange(window, dtype=np.int32)
    for j, (state, m) in enumerate(zip(states, meta)):
        node = int(m["client_id"])
        if not 0 <= node < pm.shape[1]:
            raise ValueError(f"bad client ID {node}")
        if not torch.allclose(state["prototype.prototypes"].cpu(), q, atol=1e-5, rtol=1e-5):
            raise ValueError(f"client {node} codebook differs from checkpoint global Q")
        model.load_state_dict(state, strict=True)
        model.eval()
        train_vals = pm[:split_bounds(len(pm), desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"])["train"][1], node, 0]
        mean, std = float(train_vals.mean()), float(train_vals.std())
        if std == 0:
            std = 1.0
        if not np.isclose(mean, float(m["target_mean"]), atol=1e-5):
            raise ValueError(f"client {node} normalization mean differs from checkpoint")
        if not np.isclose(std, float(m["target_std"]), atol=1e-5):
            raise ValueError(f"client {node} normalization std differs from checkpoint")
        lo, hi = j * n_windows, (j + 1) * n_windows
        result["client_id"][lo:hi] = node
        result["last_input_idx"][lo:hi] = last_input
        result["last_target_idx"][lo:hi] = last_target
        raw_x = np.asarray(pm[starts[:, None] + offset[None, :], node, 0], dtype=np.float32)
        result["pm_window"][lo:hi] = raw_x
        for name, feature in [("tp", 8), ("t2m", 2), ("humidity", 6), ("blh", 4)]:
            result[name][lo:hi] = full[last_input, node, feature]
        for name in ("hour_slot", "day_of_week", "month", "week"):
            result[name][lo:hi] = fields[name]
        w_parts = []
        with torch.inference_mode():
            for b in range(0, n_windows, batch_size):
                e = min(n_windows, b + batch_size)
                x = torch.from_numpy((raw_x[b:e] - mean) / std).unsqueeze(-1).to(device)
                t = torch.from_numpy(np.asarray(pm[last_input[b:e], node, 1:3]).copy()).to(device)
                _, _, w = model(x, temporal_id=t)
                if w is None or w.shape[1:] != (1, k):
                    raise ValueError(f"unexpected assignment shape {None if w is None else w.shape}")
                w_parts.append(w[:, 0, :].cpu().numpy())
        result["weights"][lo:hi] = np.concatenate(w_parts)
        if (j + 1) % 25 == 0 or j + 1 == len(states):
            print(f"{split}: extracted {j + 1}/{len(states)} clients", flush=True)
    if not np.isfinite(result["weights"]).all():
        raise ValueError("nonfinite prototype weights")
    if not np.allclose(result["weights"].sum(axis=1), 1, atol=1e-5):
        raise ValueError("prototype weights do not sum to one")
    return result


def control_features(rows: dict[str, np.ndarray]):
    x = rows["pm_window"].astype(np.float32)
    summary = np.column_stack([
        x, x.mean(axis=1), x.std(axis=1), x[:, -1] - x[:, 0],
        x.min(axis=1), x.max(axis=1),
    ]).astype(np.float32)
    cats = np.column_stack([
        rows["client_id"], rows["hour_slot"], rows["day_of_week"], rows["month"],
    ]).astype(np.int32)
    return summary, cats


def metric_record(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def week_block_interval(y, p0, p1, week, reps, rng):
    records = []
    for block in np.unique(week):
        ix = week == block
        if len(np.unique(y[ix])) < 2:
            continue
        records.append({
            "week": str(block), "n": int(ix.sum()),
            "auc_delta": float(roc_auc_score(y[ix], p1[ix]) - roc_auc_score(y[ix], p0[ix])),
            "brier_improvement": float(brier_score_loss(y[ix], p0[ix]) - brier_score_loss(y[ix], p1[ix])),
        })
    if len(records) < 5:
        raise ValueError("fewer than five test weeks with both weather classes")
    deltas = np.array([r["auc_delta"] for r in records])
    briers = np.array([r["brier_improvement"] for r in records])
    draw = rng.integers(0, len(records), size=(reps, len(records)))
    return {
        "n_valid_weeks": len(records),
        "mean_weekly_auc_delta": float(deltas.mean()),
        "mean_weekly_auc_delta_ci95": np.quantile(deltas[draw].mean(axis=1), [0.025, 0.975]).tolist(),
        "mean_weekly_brier_improvement": float(briers.mean()),
        "mean_weekly_brier_improvement_ci95": np.quantile(briers[draw].mean(axis=1), [0.025, 0.975]).tolist(),
        "positive_auc_weeks": int((deltas > 0).sum()),
        "weeks": records,
    }


def matched_strata(rows, train_pm_cut):
    """Coarse observed-input controls for descriptive slot-weather association."""
    q = np.searchsorted(train_pm_cut, rows["pm_window"][:, -1])
    slope_sign = (rows["pm_window"][:, -1] >= rows["pm_window"][:, 0]).astype(np.int32)
    season = ((rows["month"].astype(np.int32) - 1) // 3)
    return ((((rows["client_id"].astype(np.int32) * 5 + q) * 2
              + slope_sign) * 8 + rows["hour_slot"]) * 4 + season)


def conditional_slot_association(rows, y, train_pm_cut, reps, block_reps, rng):
    """Within-stratum difference in soft slot weight for wet versus dry windows.

    The coefficient is the fixed-effect slope of slot weight on the binary
    weather label, after removing each matched stratum's mean. Slot p-values
    use a max-|coefficient| null to account for selecting among K slots.
    Week bootstrap resamples fixed-centering contributions as a block-robust
    sensitivity interval; permutation p-values remain exploratory because
    shuffling individual windows does not preserve time autocorrelation.
    """
    group = matched_strata(rows, train_pm_cut)
    _, inverse = np.unique(group, return_inverse=True)
    counts = np.bincount(inverse)
    y = y.astype(np.float64)
    w = rows["weights"].astype(np.float64)
    y_mean = np.bincount(inverse, weights=y) / counts
    w_mean = np.column_stack([
        np.bincount(inverse, weights=w[:, k]) / counts for k in range(w.shape[1])
    ])
    yc = y - y_mean[inverse]
    wc = w - w_mean[inverse]
    denom = float(yc @ yc)
    if denom <= 0:
        raise ValueError("no mixed-weather matched strata")
    beta = (yc @ wc) / denom
    mixed = (np.bincount(inverse, weights=y) > 0) & (np.bincount(inverse, weights=y) < counts)
    matched_n = int(counts[mixed].sum())
    if matched_n < max(50, int(0.05 * len(y))):
        raise ValueError("too few matched mixed-weather windows")

    week_names, week_inverse = np.unique(rows["week"], return_inverse=True)
    weekly_numerator = np.zeros((len(week_names), w.shape[1]), dtype=np.float64)
    weekly_denominator = np.bincount(week_inverse, weights=yc * yc)
    for k in range(w.shape[1]):
        weekly_numerator[:, k] = np.bincount(
            week_inverse, weights=yc * wc[:, k], minlength=len(week_names))
    draw = rng.integers(0, len(week_names), size=(block_reps, len(week_names)))
    boot_beta = weekly_numerator[draw].sum(axis=1) / weekly_denominator[draw].sum(axis=1)[:, None]
    ci = np.quantile(boot_beta, [0.025, 0.975], axis=0).T

    order = np.argsort(group, kind="stable")
    max_null = []
    for _ in range(reps):
        source = np.lexsort((rng.random(len(group)), group))
        shuffled = np.empty_like(wc)
        shuffled[order] = wc[source]
        max_null.append(float(np.max(np.abs((yc @ shuffled) / denom))))
    max_null = np.asarray(max_null)
    return {
        "strata": "client x PM25-last training quintile x PM25 slope sign x 3-hour slot x season",
        "n_strata": int(len(counts)),
        "n_mixed_weather_strata": int(mixed.sum()),
        "n_windows_in_mixed_strata": matched_n,
        "slot_weight_wet_minus_dry_fixed_effect": beta.tolist(),
        "slot_weight_week_block_ci95": ci.tolist(),
        "max_abs_slot_effect": float(np.max(np.abs(beta))),
        "max_slot_fwer_p_exploratory": float(
            (1 + np.sum(max_null >= np.max(np.abs(beta)))) / (1 + reps)
        ) if reps else None,
        "slot_fwer_p_exploratory": [
            float((1 + np.sum(max_null >= abs(v))) / (1 + reps)) for v in beta
        ] if reps else None,
        "permutation_reps": int(reps),
        "permutation_limit": "individual-window shuffling does not preserve autocorrelation",
    }


def conditional_permutation_pvalue(rows, y, base_design, plus_model, observed_auc, reps, rng, train_pm_cut):
    """Shuffle complete slot vectors within station × PM level/trend × time × season."""
    group = matched_strata(rows, train_pm_cut)
    order = np.argsort(group, kind="stable")
    fixed = np.asarray(base_design @ plus_model.coef_[0, :base_design.shape[1]]).ravel()
    fixed += plus_model.intercept_[0]
    proto_beta = plus_model.coef_[0, base_design.shape[1]:]
    if proto_beta.shape[0] != rows["weights"].shape[1] - 1:
        raise ValueError("prototype coefficient shape mismatch")
    null_auc = []
    for _ in range(reps):
        source = np.lexsort((rng.random(len(group)), group))
        shuffled = np.empty_like(rows["weights"][:, :-1])
        shuffled[order] = rows["weights"][source, :-1]
        score = fixed + shuffled @ proto_beta
        null_auc.append(float(roc_auc_score(y, score)))
    p = (1 + sum(v >= observed_auc for v in null_auc)) / (1 + reps)
    return {"p_upper": float(p), "null_auc_mean": float(np.mean(null_auc)),
            "null_auc_q95": float(np.quantile(null_auc, 0.95)),
            "n_permutations": reps,
            "strata": "client x PM25-last quintile x PM25 slope sign x 3-hour slot x season"}


def main():
    args = args_parser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pm, full, desc, minutes = load_memmaps(args.data_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if len(checkpoint["client_models"]) != pm.shape[1]:
        raise ValueError("E1 primary checkpoint must contain all KnowAir clients")
    bounds = split_bounds(len(pm), desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"])
    thresholds = {
        "high_precip": float(np.quantile(full[:bounds["train"][1], :, 8], 0.9)),
        "high_temp": float(np.quantile(full[:bounds["train"][1], :, 2], 0.9)),
        "high_humidity": float(np.quantile(full[:bounds["train"][1], :, 6], 0.9)),
        "low_blh": float(np.quantile(full[:bounds["train"][1], :, 4], 0.1)),
    }
    if not np.isfinite(list(thresholds.values())).all():
        raise ValueError("nonfinite weather threshold")
    device = torch.device(args.device)
    model = load_model(checkpoint, args.repo, device)
    rows = {}
    for split in ("train", "test"):
        rows[split] = extract_split(split, bounds[split], pm, full, desc, minutes,
                                    checkpoint, model, device, args.batch_size,
                                    stride=args.probe_train_stride if split == "train" else 1)
        if not args.no_save_activations:
            np.savez_compressed(args.output_dir / f"activations_{split}.npz", **rows[split])

    labels = {}
    for split in ("train", "test"):
        r = rows[split]
        labels[split] = {
            "high_precip": (r["tp"] > thresholds["high_precip"]).astype(np.int8),
            "high_temp": (r["t2m"] > thresholds["high_temp"]).astype(np.int8),
            "high_humidity": (r["humidity"] > thresholds["high_humidity"]).astype(np.int8),
            "low_blh": (r["blh"] < thresholds["low_blh"]).astype(np.int8),
        }
        if not 0 < labels[split]["high_precip"].mean() < 1:
            raise ValueError(f"{split}: high-precip label is single-class")

    vnum, vcat = control_features(rows["train"])
    tnum, tcat = control_features(rows["test"])
    num_scale = StandardScaler().fit(vnum)
    cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32).fit(vcat)
    def design(num, cat):
        return sparse.hstack([
            sparse.csr_matrix(num_scale.transform(num).astype(np.float32)),
            cat_enc.transform(cat),
        ], format="csr")
    bv, bt = design(vnum, vcat), design(tnum, tcat)
    pv = sparse.hstack([bv, sparse.csr_matrix(rows["train"]["weights"][:, :-1])], format="csr")
    pt = sparse.hstack([bt, sparse.csr_matrix(rows["test"]["weights"][:, :-1])], format="csr")
    yv, yt = labels["train"]["high_precip"], labels["test"]["high_precip"]
    control = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=args.seed)
    augmented = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=args.seed)
    control.fit(bv, yv)
    augmented.fit(pv, yv)
    prob0, prob1 = control.predict_proba(bt)[:, 1], augmented.predict_proba(pt)[:, 1]
    control_metrics, augmented_metrics = metric_record(yt, prob0), metric_record(yt, prob1)
    block = week_block_interval(yt, prob0, prob1, rows["test"]["week"], args.bootstrap_reps, rng)
    pm_cut = np.quantile(rows["train"]["pm_window"][:, -1], [0.2, 0.4, 0.6, 0.8])
    conditional = conditional_slot_association(
        rows["test"], yt, pm_cut, args.permutation_reps,
        args.bootstrap_reps, rng)
    perm = conditional_permutation_pvalue(rows["test"], yt, bt, augmented,
                                           augmented_metrics["auc"], args.permutation_reps,
                                           rng, pm_cut) if args.permutation_reps else None

    nonlinear = None
    if args.nonlinear_control:
        # This control can model nonlinear functions of the full PM25 history and
        # calendar, so slot gains cannot be attributed merely to linear probes.
        def hgb_design(num, cat, weights=None):
            x = np.column_stack([num_scale.transform(num), cat.astype(np.float32)])
            return np.column_stack([x, weights[:, :-1]]) if weights is not None else x
        cat_mask = np.array([False] * vnum.shape[1] + [True] * vcat.shape[1])
        h0 = HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=31, min_samples_leaf=50,
            learning_rate=0.05, l2_regularization=1.0,
            categorical_features=cat_mask, random_state=args.seed,
        )
        h1 = HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=31, min_samples_leaf=50,
            learning_rate=0.05, l2_regularization=1.0,
            categorical_features=np.r_[cat_mask, np.zeros(rows["train"]["weights"].shape[1] - 1, dtype=bool)],
            random_state=args.seed,
        )
        h0.fit(hgb_design(vnum, vcat), yv)
        h1.fit(hgb_design(vnum, vcat, rows["train"]["weights"]), yv)
        hp0 = h0.predict_proba(hgb_design(tnum, tcat))[:, 1]
        hp1 = h1.predict_proba(hgb_design(tnum, tcat, rows["test"]["weights"]))[:, 1]
        nonlinear = {
            "control": metric_record(yt, hp0),
            "control_plus_prototypes": metric_record(yt, hp1),
            "week_block_interval": week_block_interval(
                yt, hp0, hp1, rows["test"]["week"], args.bootstrap_reps, rng),
        }

    wt = rows["test"]["weights"]
    dominant = wt.argmax(axis=1)
    per_proto = []
    for slot in range(wt.shape[1]):
        mask = dominant == slot
        entry = {
            "slot": slot, "dominant_n": int(mask.sum()),
            "dominant_fraction": float(mask.mean()),
            "mean_soft_weight": float(wt[:, slot].mean()),
        }
        for key, y in labels["test"].items():
            prevalence = float(y.mean())
            hard_rate = float(y[mask].mean()) if mask.any() else None
            soft_rate = float(np.average(y, weights=wt[:, slot]))
            entry[f"{key}_base_rate"] = prevalence
            entry[f"{key}_hard_rate"] = hard_rate
            entry[f"{key}_hard_enrichment"] = hard_rate / prevalence if hard_rate is not None else None
            entry[f"{key}_soft_rate"] = soft_rate
            entry[f"{key}_soft_enrichment"] = soft_rate / prevalence
        per_proto.append(entry)
    pd.DataFrame(per_proto).to_csv(args.output_dir / "per_prototype.csv", index=False)
    entropy = -(wt * np.log(np.maximum(wt, 1e-12))).sum(axis=1)
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_config": cfg,
        "data_dir": str(args.data_dir.resolve()),
        "split_bounds_half_open": bounds,
        "window_alignment": "absolute last input = split start + local window start + window_size - 1",
        "weather_label_time": "last input timestamp; no future weather is model input",
        "weather_thresholds_train_only": thresholds,
        "n_clients": len(checkpoint["client_models"]),
        "n_train_probe_windows": len(yv),
        "probe_train_stride": args.probe_train_stride,
        "n_test_windows": len(yt),
        "high_precip_train_rate": float(yv.mean()),
        "high_precip_test_rate": float(yt.mean()),
        "mean_assignment_entropy": float(entropy.mean()),
        "effective_slots_exp_entropy": float(np.exp(entropy.mean())),
        "linear_control": control_metrics,
        "linear_control_plus_prototypes": augmented_metrics,
        "linear_paired_week_block_interval": block,
        "matched_conditional_slot_association": conditional,
        "linear_conditional_permutation": perm,
        "nonlinear_control_comparison": nonlinear,
        "interpretation_limit": "Weather association is descriptive; it does not establish confounder identifiability or back-door adjustment.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({
        "threshold_tp_m": thresholds["high_precip"],
        "test_event_rate": float(yt.mean()),
        "linear_auc_base": control_metrics["auc"],
        "linear_auc_with_slots": augmented_metrics["auc"],
        "linear_brier_base": control_metrics["brier"],
        "linear_brier_with_slots": augmented_metrics["brier"],
        "week_auc_delta_ci95": block["mean_weekly_auc_delta_ci95"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
