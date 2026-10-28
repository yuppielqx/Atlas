"""
Align prototype activations with environmental conditions and compute correlation stats.

Inputs:
  - activations.pt (from extract_prototype_activations.py)
  - weather.csv (from fetch_weather.py)
  - sensor_weather_map.csv (from fetch_weather.py)

Outputs (saved to output_dir):
  - proto_condition_matrix.npy  : (K, n_conditions) P(condition | dominant_proto = i)
  - condition_names.txt         : list of condition names
  - panel_b_sensor.npz          : data for the timeline panel
  - panel_c_usage.npz           : per-client prototype usage + sensor selection
  - corr_metadata.json          : summary stats

Usage:
    python scripts/compute_prototype_env_correlation.py \
        --activations output/zinv_ksweep_all_k10_r200/activations_test.pt \
        --weather output/weather.csv \
        --sensor_map output/sensor_weather_map.csv \
        --output_dir output/zinv_ksweep_all_k10_r200
"""

import os
import sys
import argparse
import json
import csv
import datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute prototype-environment correlation statistics")
    parser.add_argument("--activations", type=str, required=True,
                        help="Path to activations.pt")
    parser.add_argument("--weather", type=str, required=True,
                        help="Path to weather.csv")
    parser.add_argument("--sensor_map", type=str, required=True,
                        help="Path to sensor_weather_map.csv")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_sensors_panel_c", type=int, default=4,
                        help="Number of sensors to select for Panel (c)")
    return parser.parse_args()


# ---- Condition definitions ----

def define_conditions():
    """Return list of (condition_name, is_temporal, is_weather, fn).

    fn(time_of_day, day_of_week, precip) -> bool
    """
    conditions = []

    # Temporal: time-of-day bins
    conditions.append(("Morning (6-10h)", lambda tod, dow, p: 0.25 <= tod < 0.42))
    conditions.append(("Noon (10-16h)", lambda tod, dow, p: 0.42 <= tod < 0.67))
    conditions.append(("Evening (16-20h)", lambda tod, dow, p: 0.67 <= tod < 0.83))
    conditions.append(("Night (20-6h)", lambda tod, dow, p: tod >= 0.83 or tod < 0.25))

    # Temporal: day-of-week
    conditions.append(("Weekday", lambda tod, dow, p: dow < 5.0 / 7.0))
    conditions.append(("Weekend", lambda tod, dow, p: dow >= 5.0 / 7.0))

    # Weather
    conditions.append(("Rain", lambda tod, dow, p: p is not None and p > 0))
    conditions.append(("Clear", lambda tod, dow, p: p is not None and p == 0))

    return conditions


# ---- Weather loading ----

def load_weather(weather_csv):
    """Load weather data as dict: (datetime_str, region_id) -> {temp, precip, code}."""
    weather = {}
    with open(weather_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt_str = row["datetime"]  # e.g. "2012-03-01T00:00"
            rid = int(row["region_id"])
            weather[(dt_str, rid)] = {
                "temperature_2m": float(row["temperature_2m"]) if row["temperature_2m"] else None,
                "precipitation": float(row["precipitation"]) if row["precipitation"] else 0.0,
                "weather_code": int(row["weather_code"]) if row["weather_code"] else None,
            }
    return weather


def load_sensor_weather_map(map_csv):
    """Return dict: sensor_id -> region_id."""
    mapping = {}
    with open(map_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["sensor_id"])] = int(row["region_id"])
    return mapping


# ---- Timestamp matching ----

def match_weather(tod, dow, ordinal, sensor_id, weather, sensor_region_map):
    """Find the nearest hourly weather record for a given window.

    Args:
        tod: time_of_day in [0, 1)
        dow: day_of_week in [0, 1)
        ordinal: datetime ordinal (days since 0001-01-01)
        sensor_id: sensor ID
        weather: dict of datetime_str -> weather data
        sensor_region_map: sensor_id -> region_id

    Returns:
        precipitation value (float or None)
    """
    region = sensor_region_map.get(sensor_id, 0)

    # Reconstruct approximate datetime from ordinal + time_of_day
    try:
        base_date = dt.datetime.fromordinal(int(ordinal))
        # time_of_day is in [0,1), convert to hour
        hour_float = float(tod) * 24.0
        hour = int(hour_float)
        minute = int((hour_float - hour) * 60)
        # Round minute to nearest hour for weather matching
        if minute >= 30:
            hour = (hour + 1) % 24
        window_dt = base_date.replace(hour=hour, minute=0, second=0, microsecond=0)
    except (ValueError, OverflowError):
        return None

    # Try exact match first, then search nearby hours (keyed by region)
    dt_str = window_dt.strftime("%Y-%m-%dT%H:%M")
    key = (dt_str, region)
    if key in weather:
        return weather[key]["precipitation"]

    # Search ±1 hour in the same region
    for offset in [1, -1, 2, -2]:
        try:
            alt_dt = window_dt + dt.timedelta(hours=offset)
            alt_str = alt_dt.strftime("%Y-%m-%dT%H:%M")
            alt_key = (alt_str, region)
            if alt_key in weather:
                return weather[alt_key]["precipitation"]
        except (ValueError, OverflowError):
            continue

    return None


# ---- Panel (a): Prototype-Condition Matrix ----

def compute_condition_matrix(clients, conditions, weather, sensor_region_map):
    """Compute P(condition | dominant_proto = k) for each prototype.

    Returns:
        cond_matrix: (K, n_conditions) float32
        condition_names: list of str
        proto_support: (K,) int, number of windows per prototype
    """
    n_conditions = len(conditions)
    # We'll accumulate per client and then aggregate
    # For each condition: count[proto_k, cond_j] += 1 if cond_j is True
    # Then normalize by row sum

    # Need to determine K from the activations
    K = None
    for c in clients:
        if c["weights"].shape[1] > 0:
            K = c["weights"].shape[1]
            break
    if K is None:
        raise ValueError("No valid clients found!")

    joint_counts = np.zeros((K, n_conditions), dtype=np.float64)
    proto_counts = np.zeros(K, dtype=np.float64)

    for client in clients:
        dominant = client["dominant_proto"]  # (N,)
        tod_arr = client["time_of_day"]       # (N,)
        dow_arr = client["day_of_week"]       # (N,)
        ord_arr = client["datetime_ordinal"]  # (N,)
        sensor_id = client["sensor_id"]

        N = len(dominant)
        for t in range(N):
            proto = int(dominant[t])
            tod = float(tod_arr[t]) if t < len(tod_arr) else np.nan
            dow = float(dow_arr[t]) if t < len(dow_arr) else np.nan
            ordinal = float(ord_arr[t]) if t < len(ord_arr) else np.nan

            precip = match_weather(tod, dow, ordinal, sensor_id,
                                   weather, sensor_region_map)

            proto_counts[proto] += 1

            for j, (cname, cfn) in enumerate(conditions):
                try:
                    if cfn(tod, dow, precip):
                        joint_counts[proto, j] += 1
                except (ValueError, TypeError):
                    pass

    # Normalize: P(condition | proto) = count(proto & condition) / count(proto)
    cond_matrix = np.zeros_like(joint_counts, dtype=np.float32)
    for k in range(K):
        if proto_counts[k] > 0:
            cond_matrix[k] = joint_counts[k] / proto_counts[k]

    condition_names = [c[0] for c in conditions]
    return cond_matrix, condition_names, proto_counts.astype(np.int64)


# ---- Panel (b): Timeline Data ----

def select_panel_b_sensor(clients, top_n=10):
    """Select the sensor with highest prototype diversity (entropy)."""
    best_score = -1
    best_client = None
    best_weights = None

    for client in clients:
        weights = client["weights"]  # (N, K)
        if weights.shape[0] == 0:
            continue

        # Per-window entropy, averaged
        eps = 1e-10
        ent = -np.sum(weights * np.log(weights + eps), axis=1)
        avg_entropy = float(ent.mean())

        # Also consider prototype count diversity
        dominant = client["dominant_proto"]
        proto_usage = np.bincount(dominant, minlength=weights.shape[1])
        proto_diversity = float(np.sum(proto_usage > 0))

        score = avg_entropy + 0.5 * proto_diversity
        if score > best_score:
            best_score = score
            best_client = client
            best_weights = weights

    return best_client


def prepare_timeline_data(client, weather, sensor_region_map,
                          window_size=12, horizon=12):
    """Extract a representative 2-week slice for the timeline.

    Returns dict with time series for plotting.
    """
    N = len(client["dominant_proto"])
    # Pick the middle 2 weeks of data (roughly 2016 windows for 2 weeks at 5-min)
    two_weeks = 2 * 7 * 24 * 12  # 4032 windows (5-min intervals)
    start = max(0, N // 2 - two_weeks // 2)
    end = min(N, start + two_weeks)
    start = max(0, end - two_weeks)  # ensure we get exactly two_weeks if possible

    dominant = client["dominant_proto"][start:end]
    tod = client["time_of_day"][start:end]
    dow = client["day_of_week"][start:end]
    ordinal = client["datetime_ordinal"][start:end]

    # Collect precip per window
    precip_arr = np.zeros(len(dominant), dtype=np.float32)
    for t in range(len(dominant)):
        p = match_weather(float(tod[t]), float(dow[t]), float(ordinal[t]),
                          client["sensor_id"], weather, sensor_region_map)
        precip_arr[t] = p if p is not None else 0.0

    return {
        "client_id": client["client_id"],
        "sensor_id": client["sensor_id"],
        "dominant_proto": dominant,
        "time_of_day": tod,
        "day_of_week": dow,
        "precipitation": precip_arr,
        "datetime_ordinal": ordinal,
        "start_idx": start,
        "n_windows": len(dominant),
    }


# ---- Panel (c): Cross-Client Usage ----

def select_diverse_sensors(clients, n_sensors=4):
    """Select sensors with diverse prototype usage patterns.

    Strategy: cluster clients by their prototype usage distribution,
    then pick the most "typical" client from each cluster.
    """
    from sklearn.cluster import KMeans

    # Collect per-client average prototype weights
    usage_vectors = []
    valid_clients = []
    for client in clients:
        w = client["weights"]
        if w.shape[0] < 50:
            continue
        avg_w = w.mean(axis=0)  # (K,)
        usage_vectors.append(avg_w)
        valid_clients.append(client)

    if len(valid_clients) < n_sensors:
        return valid_clients

    X = np.stack(usage_vectors, axis=0)  # (N_valid, K)
    k = min(n_sensors, len(valid_clients))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Pick the client closest to each cluster centroid
    selected = []
    for r in range(k):
        mask = labels == r
        if not mask.any():
            continue
        # Pick the one closest to centroid
        cluster_X = X[mask]
        centroid = cluster_X.mean(axis=0)
        dists = np.linalg.norm(cluster_X - centroid, axis=1)
        best_idx = np.where(mask)[0][np.argmin(dists)]
        selected.append(valid_clients[best_idx])

    return selected


def compute_panel_c_data(selected_clients):
    """Compute average prototype usage distribution for selected sensors.

    Returns:
        list of dicts with {client_id, sensor_id, avg_weights (K,), dominant_proto_counts (K,)}
    """
    results = []
    for client in selected_clients:
        weights = client["weights"]
        avg_weights = weights.mean(axis=0)  # (K,)
        dominant = client["dominant_proto"]
        proto_counts = np.bincount(dominant, minlength=weights.shape[1])
        proto_freq = proto_counts.astype(np.float32) / max(proto_counts.sum(), 1)

        results.append({
            "client_id": client["client_id"],
            "sensor_id": client["sensor_id"],
            "avg_weights": avg_weights.astype(np.float32),
            "proto_freq": proto_freq.astype(np.float32),
            "n_windows": int(len(dominant)),
        })
    return results


# ---- Main ----

def main():
    args = parse_args()

    # 1. Load activations
    print(f"Loading activations from {args.activations}...")
    data = torch_load(args.activations)
    clients = data["clients"]
    K = data["K"]
    print(f"  {len(clients)} clients, K={K}")

    # 2. Load weather
    print(f"Loading weather from {args.weather}...")
    weather = load_weather(args.weather)
    print(f"  {len(weather)} weather records")

    # 3. Load sensor -> region map
    print(f"Loading sensor map from {args.sensor_map}...")
    sensor_region_map = load_sensor_weather_map(args.sensor_map)
    print(f"  {len(sensor_region_map)} sensor mappings")

    # 4. Panel (a): Condition matrix
    print("\n=== Panel (a): Prototype-Condition Matrix ===")
    conditions = define_conditions()
    cond_matrix, condition_names, proto_support = compute_condition_matrix(
        clients, conditions, weather, sensor_region_map)

    print(f"  Matrix shape: {cond_matrix.shape}")
    print(f"  Conditions: {condition_names}")
    print(f"  Proto support (total windows per proto): {proto_support}")
    print(f"  Row sums (should be ~1.0): {cond_matrix.sum(axis=1)}")

    # 5. Panel (b): Timeline sensor
    print("\n=== Panel (b): Timeline Sensor ===")
    panel_b_client = select_panel_b_sensor(clients)
    if panel_b_client is None:
        print("  WARNING: No suitable sensor found!")
        timeline = None
    else:
        print(f"  Selected sensor {panel_b_client['sensor_id']} "
              f"(client {panel_b_client['client_id']})")
        timeline = prepare_timeline_data(panel_b_client, weather, sensor_region_map)
        unique_protos = len(np.unique(timeline["dominant_proto"]))
        print(f"  Timeline: {timeline['n_windows']} windows, "
              f"{unique_protos} unique prototypes")

    # 6. Panel (c): Cross-client usage
    print("\n=== Panel (c): Cross-Client Prototype Usage ===")
    selected_clients = select_diverse_sensors(clients, args.n_sensors_panel_c)
    print(f"  Selected {len(selected_clients)} sensors:")
    for c in selected_clients:
        print(f"    Sensor {c['sensor_id']} (client {c['client_id']}): "
              f"{c['weights'].shape[0]} windows")
    panel_c_data = compute_panel_c_data(selected_clients)

    # 7. Save all outputs
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, "proto_condition_matrix.npy"), cond_matrix)
    with open(os.path.join(args.output_dir, "condition_names.txt"), "w") as f:
        for name in condition_names:
            f.write(name + "\n")

    if timeline is not None:
        np.savez(os.path.join(args.output_dir, "panel_b_sensor.npz"), **timeline)

    panel_c_path = os.path.join(args.output_dir, "panel_c_usage.npz")
    np.savez(panel_c_path,
             client_ids=np.array([c["client_id"] for c in panel_c_data]),
             sensor_ids=np.array([c["sensor_id"] for c in panel_c_data]),
             avg_weights=np.stack([c["avg_weights"] for c in panel_c_data]),
             proto_freq=np.stack([c["proto_freq"] for c in panel_c_data]),
             n_windows=np.array([c["n_windows"] for c in panel_c_data]))

    # Save metadata
    metadata = {
        "K": int(K),
        "n_clients": len(clients),
        "condition_names": condition_names,
        "proto_support": proto_support.tolist(),
        "panel_b_sensor_id": int(timeline["sensor_id"]) if timeline else None,
        "panel_c_sensor_ids": [int(c["sensor_id"]) for c in panel_c_data],
    }
    with open(os.path.join(args.output_dir, "corr_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll outputs saved to: {args.output_dir}")


def torch_load(path):
    """Load activations.pt using torch if available, else numpy fallback."""
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    except ImportError:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    main()
