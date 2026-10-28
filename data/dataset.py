"""
Data utilities for federated spatio-temporal forecasting.

Supports:
- Single-node time series (FUELS-compatible: each client = one node)
- Sliding window input-output pair generation
- METR-LA real-world traffic dataset loading
"""

import json
import os
import pickle
import numpy as np


def sliding_window(time_series, window_size, horizon):
    """Generate input-output pairs via sliding window.

    Args:
        time_series: (T,) or (T, D) numpy array
        window_size: input sequence length
        horizon: output sequence length

    Returns:
        X: (num_samples, window_size, D) inputs
        Y: (num_samples, horizon, D) targets
    """
    if time_series.ndim == 1:
        time_series = time_series[:, None]
    T, D = time_series.shape
    X, Y = [], []
    for i in range(T - window_size - horizon + 1):
        X.append(time_series[i : i + window_size])
        Y.append(time_series[i + window_size : i + window_size + horizon])
    return np.array(X), np.array(Y)


def sliding_window_temporal_identity(timestamps, window_size, horizon):
    """Return last-input-step [time_of_day, day_of_week] for each window."""
    temporal_ids = []
    for i in range(len(timestamps) - window_size - horizon + 1):
        temporal_ids.append(timestamps[i + window_size - 1])
    return np.array(temporal_ids)


# ---------------------------------------------------------------------------
# METR-LA real-world data loading
# ---------------------------------------------------------------------------

def load_metrla_raw(dataset_name="METR-LA", data_dir="datasets"):
    """Load raw METR-LA data from BasicTS memmap format.

    Reads data.dat [L, N, 3] and desc.json, returns values and timestamps.
    Channel 0 = traffic speed, channel 1 = time_of_day, channel 2 = day_of_week.

    Returns:
        values: (L, N) float32 array of traffic speeds
        timestamps: (L, 2) float32 array of [time_of_day, day_of_week]
    """
    ds_dir = os.path.join(data_dir, dataset_name)
    desc_path = os.path.join(ds_dir, "desc.json")
    dat_path = os.path.join(ds_dir, "data.dat")

    if not (os.path.exists(desc_path) and os.path.exists(dat_path)):
        raise FileNotFoundError(
            f"METR-LA data not found at {ds_dir}. "
            f"Expected data.dat and desc.json."
        )

    with open(desc_path) as f:
        desc = json.load(f)

    shape = tuple(desc["shape"])  # [L, N, 3]
    data = np.memmap(dat_path, dtype=np.float32, mode="r", shape=shape)

    values = np.array(data[:, :, 0])          # [L, N]
    timestamps = np.array(data[:, 0, 1:3])    # [L, 2]

    return values, timestamps


def load_adjacency(dataset_name="METR-LA", data_dir="datasets"):
    """Load a BasicTS-style adjacency matrix if the dataset ships one."""
    path = os.path.join(data_dir, dataset_name, "adj_mx.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, (tuple, list)):
        # BasicTS/LibCity commonly stores (sensor_ids, sensor_id_to_ind, adj).
        for item in reversed(obj):
            if hasattr(item, "shape") and len(item.shape) == 2:
                return np.asarray(item, dtype=np.float32)
        return None
    if hasattr(obj, "shape") and len(obj.shape) == 2:
        return np.asarray(obj, dtype=np.float32)
    return None


# ---------------------------------------------------------------------------
# Multi-feature data loading (for causal discovery with weather→PM2.5 etc.)
# ---------------------------------------------------------------------------

def load_multifeature_raw(dataset_name="KnowAir-Full", data_dir="datasets"):
    """Load a multi-feature BasicTS dataset [L, N, F].

    Returns:
        all_features: (L, N, F) float32 array of all channels
        timestamps:   (L, 2) float32 array of [time_of_day, day_of_week]
        desc:         dict from desc.json
    """
    ds_dir = os.path.join(data_dir, dataset_name)
    desc_path = os.path.join(ds_dir, "desc.json")
    dat_path = os.path.join(ds_dir, "data.dat")

    if not (os.path.exists(desc_path) and os.path.exists(dat_path)):
        raise FileNotFoundError(
            f"Dataset not found at {ds_dir}. Expected data.dat and desc.json."
        )

    with open(desc_path) as f:
        desc = json.load(f)

    shape = tuple(desc["shape"])  # [L, N, F]
    data = np.memmap(dat_path, dtype=np.float32, mode="r", shape=shape)
    all_features = np.array(data)  # [L, N, F]

    # Build time features from metadata
    L = shape[0]
    freq_min = desc.get("frequency_minutes", desc.get("frequency (minutes)", 5))
    from datetime import datetime, timedelta, timezone
    start_str = desc.get("start_time_utc", "2015-01-01T00:00:00+00:00")
    start = datetime.fromisoformat(start_str)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    timestamps = np.empty((L, 2), dtype=np.float32)
    for i in range(L):
        ts = start + timedelta(minutes=i * freq_min)
        timestamps[i, 0] = (ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400.0
        timestamps[i, 1] = ts.weekday() / 7.0

    return all_features, timestamps, desc


def generate_multifeature_clients(
    n_clients=None,
    client_ids=None,
    window_size=12,
    horizon=12,
    dataset_name="KnowAir-Full",
    data_dir="datasets",
    seed=42,
    target_index=-1,
    input_feature_indices=None,
    include_target_in_input=True,
):
    """Generate per-node clients from a multi-feature dataset.

    Each node becomes one federated client. Input features are used
    to predict the target. Chronological train/val/test split.

    Args:
        n_clients: number of clients (nodes). None = all nodes.
        client_ids: explicit node ids; overrides n_clients.
        window_size: input sequence length
        horizon: prediction horizon
        dataset_name: dataset folder name
        data_dir: root data directory
        seed: random seed
        target_index: which channel is the target (default: -1 = last channel)
        input_feature_indices: which channels to use as input features.
            None = all channels except target_index.
        include_target_in_input: if True, prepend the target to input features
            so the model gets autoregressive signal + exogenous weather drivers.

    Returns:
        clients_data: list of dicts with X_*, Y_*, T_*, target_mean, target_std
    """
    all_features, timestamps, desc = load_multifeature_raw(dataset_name, data_dir)
    num_nodes = all_features.shape[1]
    num_features = all_features.shape[2]

    if target_index < 0:
        target_index = num_features + target_index

    if input_feature_indices is None:
        input_feature_indices = [i for i in range(num_features) if i != target_index]

    input_dim = len(input_feature_indices)

    # Node selection
    if client_ids is not None:
        node_ids = np.asarray(client_ids, dtype=int)
        if node_ids.ndim != 1 or node_ids.size == 0:
            raise ValueError("client_ids must be a non-empty 1D list of node ids")
        bad = node_ids[(node_ids < 0) | (node_ids >= num_nodes)]
        if bad.size:
            raise ValueError(
                f"client_ids out of range for {dataset_name}: {bad.tolist()}"
            )
    else:
        rng = np.random.default_rng(seed)
        if n_clients is not None and n_clients < num_nodes:
            node_ids = rng.choice(num_nodes, n_clients, replace=False)
            node_ids = np.sort(node_ids)
        else:
            node_ids = np.arange(num_nodes)

    # Chronological split
    ratios = desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"]
    L = all_features.shape[0]
    train_end = int(L * ratios[0])
    val_end = train_end + int(L * ratios[1])

    clients_data = []

    for node_id in node_ids:
        # Extract input features and target for this node
        weather_input = all_features[:, node_id, input_feature_indices]  # [L, n_weather]
        node_target = all_features[:, node_id, target_index]             # [L,]

        # Optionally prepend target as the first input channel so the model
        # gets autoregressive signal alongside exogenous weather drivers.
        if include_target_in_input:
            node_input = np.column_stack(
                [node_target.reshape(-1, 1), weather_input]
            )  # [L, 1 + n_weather]
        else:
            node_input = weather_input  # [L, n_weather]

        # Split chronologically
        train_input = node_input[:train_end]
        val_input = node_input[train_end:val_end]
        test_input = node_input[val_end:]

        train_target = node_target[:train_end]
        val_target = node_target[train_end:val_end]
        test_target = node_target[val_end:]

        # Per-channel z-score normalization on input features (train stats)
        input_mean = np.mean(train_input, axis=0, keepdims=True)
        input_std = np.std(train_input, axis=0, keepdims=True)
        input_std[input_std == 0] = 1.0

        def _normalize_input(arr):
            return (arr - input_mean) / input_std

        train_input_n = _normalize_input(train_input)
        val_input_n = _normalize_input(val_input)
        test_input_n = _normalize_input(test_input)

        # Target normalization
        target_mean = float(np.mean(train_target))
        target_std = float(np.std(train_target))
        if target_std == 0:
            target_std = 1.0

        train_target_n = (train_target - target_mean) / target_std
        val_target_n = (val_target - target_mean) / target_std
        test_target_n = (test_target - target_mean) / target_std

        # Sliding windows: X from input features, Y from target
        X_train, _ = sliding_window(train_input_n, window_size, horizon)
        _, Y_train_raw = sliding_window(train_target_n.reshape(-1, 1), window_size, horizon)
        X_val, _ = sliding_window(val_input_n, window_size, horizon)
        _, Y_val_raw = sliding_window(val_target_n.reshape(-1, 1), window_size, horizon)
        X_test, _ = sliding_window(test_input_n, window_size, horizon)
        _, Y_test_raw = sliding_window(test_target_n.reshape(-1, 1), window_size, horizon)

        # Temporal identity features
        T_train = sliding_window_temporal_identity(timestamps[:train_end], window_size, horizon)
        T_val = sliding_window_temporal_identity(timestamps[train_end:val_end], window_size, horizon)
        T_test = sliding_window_temporal_identity(timestamps[val_end:], window_size, horizon)

        clients_data.append({
            "client_id": int(node_id),
            "X_train": X_train.astype(np.float32),
            "Y_train": Y_train_raw.astype(np.float32),
            "T_train": T_train.astype(np.float32),
            "X_val": X_val.astype(np.float32),
            "Y_val": Y_val_raw.astype(np.float32),
            "T_val": T_val.astype(np.float32),
            "X_test": X_test.astype(np.float32),
            "Y_test": Y_test_raw.astype(np.float32),
            "T_test": T_test.astype(np.float32),
            "target_mean": target_mean,
            "target_std": target_std,
            "input_feature_indices": input_feature_indices,
            "target_index": target_index,
        })

    return clients_data


def _neighbor_mean_series(values, adj, node_id, self_weight=1.0):
    """Weighted mean of a node's graph neighbors, used by graph baselines."""
    if adj is None:
        return values[:, node_id]
    weights = np.asarray(adj[node_id], dtype=np.float32).copy()
    if weights.ndim != 1 or weights.size != values.shape[1]:
        return values[:, node_id]
    weights[node_id] = max(float(weights[node_id]), float(self_weight))
    denom = float(weights.sum())
    if denom <= 0:
        return values[:, node_id]
    weights = weights / denom
    return values @ weights


def generate_metrla_clients(
    n_clients=None,
    client_ids=None,
    window_size=12,
    horizon=3,
    use_time_features=True,
    dataset_name="METR-LA",
    data_dir="datasets",
    seed=42,
    graph_context=False,
):
    """Generate per-node client datasets from METR-LA traffic data.

    Each sensor node becomes one federated client. Chronological
    train/val/test split follows the ratios in desc.json.

    Args:
        n_clients: number of clients (sensors) to use. None = all nodes.
                   If < all nodes, randomly subsamples nodes.
        client_ids: explicit sensor ids to use. When supplied, this takes
                    precedence over n_clients and preserves the given order.
        window_size: input sequence length
        horizon: prediction horizon
        use_time_features: include time_of_day and day_of_week as input features
        dataset_name: dataset folder name
        data_dir: root data directory
        seed: random seed for subsampling nodes

    Returns:
        clients_data: list of dicts with keys
            (X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler, node_id)
    """
    values, timestamps = load_metrla_raw(dataset_name, data_dir)
    adj = load_adjacency(dataset_name, data_dir) if graph_context else None
    num_nodes = values.shape[1]  # 207 for METR-LA

    # Explicit node ids are useful for nested source-subset transfer studies.
    if client_ids is not None:
        node_ids = np.asarray(client_ids, dtype=int)
        if node_ids.ndim != 1 or node_ids.size == 0:
            raise ValueError("client_ids must be a non-empty 1D list of node ids")
        if len(np.unique(node_ids)) != len(node_ids):
            raise ValueError("client_ids contains duplicates")
        bad = node_ids[(node_ids < 0) | (node_ids >= num_nodes)]
        if bad.size:
            raise ValueError(
                f"client_ids out of range for {dataset_name}: {bad.tolist()} "
                f"(valid 0..{num_nodes - 1})"
            )
        n_clients = len(node_ids)
    else:
        rng = np.random.default_rng(seed)
        if n_clients is not None and n_clients < num_nodes:
            node_ids = rng.choice(num_nodes, n_clients, replace=False)
            node_ids = np.sort(node_ids)
        else:
            node_ids = np.arange(num_nodes)
            n_clients = num_nodes

    # Chronological split ratios from desc.json
    with open(os.path.join(data_dir, dataset_name, "desc.json")) as f:
        desc = json.load(f)
    ratios = desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"]  # [0.7, 0.1, 0.2]
    L = values.shape[0]
    train_end = int(L * ratios[0])
    val_end = train_end + int(L * ratios[1])

    clients_data = []

    for node_id in node_ids:
        node_series = values[:, node_id]  # (L,)

        # Split chronologically
        train_vals = node_series[:train_end]
        val_vals = node_series[train_end:val_end]
        test_vals = node_series[val_end:]

        neighbor_series = _neighbor_mean_series(values, adj, int(node_id)) if graph_context else None

        if use_time_features:
            # Stack value + timestamps as features: (L, 3)
            train_ts = timestamps[:train_end]
            val_ts = timestamps[train_end:val_end]
            test_ts = timestamps[val_end:]

            cols_train = [train_vals]
            cols_val = [val_vals]
            cols_test = [test_vals]
            if graph_context:
                cols_train.append(neighbor_series[:train_end])
                cols_val.append(neighbor_series[train_end:val_end])
                cols_test.append(neighbor_series[val_end:])
            train_data = np.column_stack(cols_train + [train_ts])
            val_data = np.column_stack(cols_val + [val_ts])
            test_data = np.column_stack(cols_test + [test_ts])
        else:
            if graph_context:
                train_data = np.column_stack([train_vals, neighbor_series[:train_end]])
                val_data = np.column_stack([val_vals, neighbor_series[train_end:val_end]])
                test_data = np.column_stack([test_vals, neighbor_series[val_end:]])
            else:
                train_data = train_vals.reshape(-1, 1)
                val_data = val_vals.reshape(-1, 1)
                test_data = test_vals.reshape(-1, 1)

        # Z-score normalize on the target (channel 0) using train stats
        train_mean = float(np.mean(train_vals))
        train_std = float(np.std(train_vals))
        if train_std == 0:
            train_std = 1.0

        def _normalize(arr):
            out = arr.copy()
            out[:, 0] = (out[:, 0] - train_mean) / train_std
            if graph_context and out.shape[1] > 1:
                neigh_train = neighbor_series[:train_end]
                neigh_mean = float(np.mean(neigh_train))
                neigh_std = float(np.std(neigh_train)) or 1.0
                out[:, 1] = (out[:, 1] - neigh_mean) / neigh_std
            return out

        train_data = _normalize(train_data)
        val_data = _normalize(val_data)
        test_data = _normalize(test_data)

        # Sliding windows
        X_train, Y_train = sliding_window(train_data, window_size, horizon)
        X_val, Y_val = sliding_window(val_data, window_size, horizon)
        X_test, Y_test = sliding_window(test_data, window_size, horizon)

        # STID temporal identity is passed separately so value inputs can stay
        # one-dimensional instead of appending time features to X.
        T_train = sliding_window_temporal_identity(timestamps[:train_end], window_size, horizon)
        T_val = sliding_window_temporal_identity(timestamps[train_end:val_end], window_size, horizon)
        T_test = sliding_window_temporal_identity(timestamps[val_end:], window_size, horizon)

        clients_data.append({
            "client_id": int(node_id),
            "X_train": X_train.astype(np.float32),
            "Y_train": Y_train.astype(np.float32),
            "T_train": T_train.astype(np.float32),
            "X_val": X_val.astype(np.float32),
            "Y_val": Y_val.astype(np.float32),
            "T_val": T_val.astype(np.float32),
            "X_test": X_test.astype(np.float32),
            "Y_test": Y_test.astype(np.float32),
            "T_test": T_test.astype(np.float32),
            "target_mean": train_mean,
            "target_std": train_std,
        })

    return clients_data


def generate_centralized_data(
    n_clients=None,
    window_size=12,
    horizon=3,
    dataset_name="METR-LA",
    data_dir="datasets",
    seed=42,
):
    """Pooled, globally-normalized data for the centralized baseline.

    Unlike ``generate_metrla_clients`` (per-node z-score, one Client per node),
    this pools all selected nodes into a single training set normalized by ONE
    global scaler (mean/std over all selected nodes' train targets), as is
    standard for centralized training. Test data is kept per-node so the same
    per-node macro metrics can be reported.

    Returns a dict with:
        X_train, Y_train, T_train : pooled windows across all nodes (float32)
        X_val,   Y_val,   T_val   : pooled validation windows
        test_nodes : list of {client_id, X_test, Y_test, T_test} (one per node)
        target_mean, target_std   : the single global scaler
        node_ids                  : list of node indices used
    """
    values, timestamps = load_metrla_raw(dataset_name, data_dir)
    num_nodes = values.shape[1]

    rng = np.random.default_rng(seed)
    if n_clients is not None and n_clients < num_nodes:
        node_ids = np.sort(rng.choice(num_nodes, n_clients, replace=False))
    else:
        node_ids = np.arange(num_nodes)

    with open(os.path.join(data_dir, dataset_name, "desc.json")) as f:
        desc = json.load(f)
    ratios = desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"]
    L = values.shape[0]
    train_end = int(L * ratios[0])
    val_end = train_end + int(L * ratios[1])

    # Single global scaler over all selected nodes' train target values.
    train_block = values[:train_end][:, node_ids]
    g_mean = float(np.mean(train_block))
    g_std = float(np.std(train_block))
    if g_std == 0:
        g_std = 1.0

    Xtr, Ytr, Ttr, Xva, Yva, Tva = [], [], [], [], [], []
    test_nodes = []
    for node_id in node_ids:
        series = ((values[:, node_id] - g_mean) / g_std).reshape(-1, 1)
        tr, va, te = series[:train_end], series[train_end:val_end], series[val_end:]

        x_tr, y_tr = sliding_window(tr, window_size, horizon)
        x_va, y_va = sliding_window(va, window_size, horizon)
        x_te, y_te = sliding_window(te, window_size, horizon)
        t_tr = sliding_window_temporal_identity(timestamps[:train_end], window_size, horizon)
        t_va = sliding_window_temporal_identity(timestamps[train_end:val_end], window_size, horizon)
        t_te = sliding_window_temporal_identity(timestamps[val_end:], window_size, horizon)

        Xtr.append(x_tr); Ytr.append(y_tr); Ttr.append(t_tr)
        Xva.append(x_va); Yva.append(y_va); Tva.append(t_va)
        test_nodes.append({
            "client_id": int(node_id),
            "X_test": x_te.astype(np.float32),
            "Y_test": y_te.astype(np.float32),
            "T_test": t_te.astype(np.float32),
        })

    return {
        "X_train": np.concatenate(Xtr).astype(np.float32),
        "Y_train": np.concatenate(Ytr).astype(np.float32),
        "T_train": np.concatenate(Ttr).astype(np.float32),
        "X_val": np.concatenate(Xva).astype(np.float32),
        "Y_val": np.concatenate(Yva).astype(np.float32),
        "T_val": np.concatenate(Tva).astype(np.float32),
        "test_nodes": test_nodes,
        "target_mean": g_mean,
        "target_std": g_std,
        "node_ids": node_ids.tolist(),
    }
