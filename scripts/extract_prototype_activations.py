"""
Extract per-window prototype activations from a trained CausalFedST checkpoint.

For each client's test set, runs a forward pass and collects:
  - dominant_proto[t]: argmax prototype index per window
  - weights[t, :]: softmax assignment weights (K-dimensional) per window
  - timestamps[t]: (time_of_day, day_of_week, datetime_ordinal) per window
  - client_id, sensor_id

Output: activations.pt with structure:
  {
    "clients": [
      {
        "client_id": int,
        "sensor_id": int,
        "dominant_proto": (N_windows,) int64,
        "weights": (N_windows, K) float32,
        "time_of_day": (N_windows,) float32,
        "day_of_week": (N_windows,) float32,
        "datetime_ordinal": (N_windows,) float64,
      },
      ...
    ],
    "K": int,
    "tau": float,
    "config": dict,
  }

Usage:
    python scripts/extract_prototype_activations.py \
        --checkpoint output/zinv_ksweep_all_k10_r200/checkpoint.pt \
        --data_split test \
        --device cuda:0
"""

import os
import sys
import argparse
import datetime as dt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import generate_metrla_clients
from models.causal_fedst import CausalFedST
from models.encoders import build_encoder_kwargs

DATASET_DIRS = {
    "metr-la": "METR-LA",
    "pems-bay": "PEMS-BAY",
    "pems03": "PEMS03",
    "pems04": "PEMS04",
    "knowair-pm25": "KnowAir-PM25",
}

# METR-LA starts 2012-03-01 00:00 (Thursday), 5-minute intervals.
# This is the canonical release date from Li et al. (ICLR 2018).
METRLA_START = dt.datetime(2012, 3, 1, 0, 0, 0)
METRLA_FREQ_MIN = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-window prototype activations from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint.pt")
    parser.add_argument("--data_split", type=str, default="test",
                        choices=["val", "test", "train"])
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: checkpoint dir)")
    parser.add_argument("--max_clients", type=int, default=None,
                        help="Limit number of clients (for debugging)")
    return parser.parse_args()


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def build_model_from_config(config, device):
    encoder_type = config.get("encoder_type", "stid")
    encoder_kwargs = build_encoder_kwargs(encoder_type, config.get("encoder_opt", []))
    use_prototype = config.get("fl_mode", "causal_proto") not in {
        "local", "fedavg", "fedavg_enc", "ditto"}
    use_residual = config.get("use_residual", True)
    residual_mode = config.get("residual_mode", "fixed") if use_residual else "none"

    model = CausalFedST(
        input_dim=config.get("input_dim", 1)
                  if not config.get("graph_context", False) else 2,
        hid_dim=config.get("hid_dim", 64),
        seq_len=config.get("window_size", 12),
        horizon=config.get("horizon", 12),
        K=config.get("K", 10),
        tau=config.get("tau", 0.1),
        stid_num_layers=config.get("stid_num_layers", 3),
        dropout=config.get("dropout", 0.1),
        use_residual=use_residual,
        residual_mode=residual_mode,
        residual_scale=config.get("residual_scale", 1.0),
        residual_gate_init=config.get("residual_gate_init", -2.0),
        encoder_type=encoder_type,
        encoder_kwargs=encoder_kwargs,
        use_prototype=use_prototype,
        use_aux_inv=(config.get("aux_inv_weight", 0.0) > 0),
        separate_residual=config.get("separate_residual", False),
    )
    return model


def time_features_to_datetime(tod, dow, window_idx, start_dt=METRLA_START, freq_min=METRLA_FREQ_MIN):
    """Convert [0,1]-normalized time_of_day and day_of_week to absolute datetime.

    Args:
        tod: time_of_day in [0, 1), where 0 = 00:00, 0.5 = 12:00
        dow: day_of_week in [0, 1), where 0 = Monday (approx)
        window_idx: index of the window (for absolute day calculation)
        start_dt: absolute start datetime of the dataset
        freq_min: time step frequency in minutes

    Returns:
        datetime object
    """
    # Reconstruct absolute time from window index
    # Each window covers window_size input steps + (horizon - 1) future steps
    # The temporal identity is at the last input step
    # We use window_idx + window_size - 1 as the offset (last input step)
    # But since we don't have the exact window_size here, we approximate
    # using the absolute ordinal date from window_idx
    total_minutes = window_idx * freq_min
    abs_dt = start_dt + dt.timedelta(minutes=float(total_minutes))
    return abs_dt


def collect_activations(checkpoint_path, args):
    """For each client, collect per-window prototype weights and timestamps."""
    device = torch.device(args.device)
    ckpt = load_checkpoint(checkpoint_path, device)
    config = ckpt["config"]
    client_models = ckpt["client_models"]
    global_prototypes = ckpt["server_prototypes"].float().to(device)
    K = config.get("K", 10)
    tau = config.get("tau", 0.1)

    print(f"K={K}, tau={tau}, clients={len(client_models)}, "
          f"prototypes shape={tuple(global_prototypes.shape)}")

    dataset_name = config.get("dataset", "metr-la")
    dataset_dir = DATASET_DIRS[dataset_name]
    window_size = config.get("window_size", 12)
    horizon = config.get("horizon", 12)
    graph_context = config.get("graph_context", False)
    client_ids_file = config.get("client_ids_file", None)

    # Read explicit client ids if used
    explicit_ids = None
    if client_ids_file:
        import re
        with open(client_ids_file) as f:
            text = f.read()
        toks = [t for t in re.split(r"[\s,]+", text.strip()) if t]
        explicit_ids = [int(t) for t in toks]

    n_clients = len(client_models)
    if explicit_ids is not None:
        n_clients = len(explicit_ids)

    print(f"Loading dataset {dataset_dir} with {n_clients} clients...")
    clients_data = generate_metrla_clients(
        n_clients=n_clients if explicit_ids is None else None,
        client_ids=explicit_ids,
        window_size=window_size,
        horizon=horizon,
        use_time_features=False,
        dataset_name=dataset_dir,
        data_dir=args.data_dir,
        seed=config.get("seed", 42),
        graph_context=graph_context,
    )
    print(f"  Loaded {len(clients_data)} clients")

    # Load sensor locations to get sensor_id
    import csv
    sensor_id_map = {}  # client_index -> sensor_id
    loc_csv = os.path.join(args.data_dir, dataset_dir, "graph_sensor_locations.csv")
    if os.path.exists(loc_csv):
        with open(loc_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["index"])
                sid = int(row["sensor_id"])
                sensor_id_map[idx] = sid

    # Build model template
    model = build_model_from_config(config, device)
    model.set_prototypes(global_prototypes)
    model.eval()

    from torch.utils.data import DataLoader, TensorDataset

    all_clients = []
    n = min(len(clients_data), len(client_models))
    if args.max_clients:
        n = min(n, args.max_clients)

    for i in range(n):
        sd = client_models[i]
        model.load_state_dict(sd, strict=False)
        model.to(device)

        data = clients_data[i]
        if args.data_split == "train":
            X = data["X_train"]
            T = data.get("T_train")
        elif args.data_split == "test":
            X = data["X_test"]
            T = data.get("T_test")
        else:
            X = data["X_val"] if data.get("X_val") is not None else data["X_train"]
            T = data.get("T_val") if data.get("T_val") is not None else data.get("T_train")

        X_t = torch.from_numpy(X).float()
        T_t = torch.from_numpy(T).float() if T is not None else None

        if T_t is not None:
            loader = DataLoader(TensorDataset(X_t, T_t),
                                batch_size=args.batch_size, shuffle=False)
        else:
            loader = DataLoader(TensorDataset(X_t),
                                batch_size=args.batch_size, shuffle=False)

        client_id = int(data.get("client_id", i))
        sensor_id = sensor_id_map.get(client_id, client_id)

        # Collect per-window results
        all_dominant = []
        all_weights = []
        all_tod = []
        all_dow = []
        all_ordinal = []

        window_offset = 0  # track window index for datetime reconstruction
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(device)
                batch_t = batch[1].to(device) if len(batch) > 1 else None

                z = model.encoder(batch_x, temporal_id=batch_t)
                B_, T_, D_ = z.shape
                z_flat = z.reshape(B_ * T_, D_)
                z_norm = F.normalize(z_flat, dim=-1)
                p_norm = F.normalize(global_prototypes, dim=-1)
                sim = z_norm @ p_norm.T
                w = F.softmax(sim / tau, dim=-1)  # (B*T, K)

                w_np = w.cpu().numpy()
                dominant = np.argmax(w_np, axis=1)  # (B*T,)

                all_dominant.append(dominant)
                all_weights.append(w_np)

                # Extract per-window temporal features
                if batch_t is not None:
                    # batch_t shape: (B, 2) from sliding_window_temporal_identity
                    # Each row is [time_of_day, day_of_week] for the last input step
                    bt_np = batch_t.cpu().numpy()
                    all_tod.append(bt_np[:, 0])
                    all_dow.append(bt_np[:, 1])

                    # Compute datetime ordinals for each window
                    for j in range(B_):
                        abs_dt = time_features_to_datetime(
                            float(bt_np[j, 0]), float(bt_np[j, 1]),
                            window_offset + j)
                        all_ordinal.append(abs_dt.toordinal())
                else:
                    all_tod.append(np.full(B_, np.nan))
                    all_dow.append(np.full(B_, np.nan))
                    all_ordinal.append(np.full(B_, np.nan))

                window_offset += B_

        dominant_proto = np.concatenate(all_dominant, axis=0)
        weights_mat = np.concatenate(all_weights, axis=0)

        if all_tod:
            tod_arr = np.concatenate(all_tod, axis=0) if all_tod[0].ndim == 1 else np.concatenate(all_tod, axis=0)
            dow_arr = np.concatenate(all_dow, axis=0) if all_dow[0].ndim == 1 else np.concatenate(all_dow, axis=0)
        else:
            tod_arr = np.array([])
            dow_arr = np.array([])

        # Handle ordinal array
        if len(all_ordinal) > 0 and not np.isscalar(all_ordinal[0]):
            ordinal_arr = np.array(all_ordinal)
        else:
            ordinal_arr = np.array(all_ordinal) if all_ordinal else np.array([])

        client_entry = {
            "client_id": client_id,
            "sensor_id": sensor_id,
            "dominant_proto": dominant_proto.astype(np.int64),
            "weights": weights_mat.astype(np.float32),
            "time_of_day": tod_arr.astype(np.float32),
            "day_of_week": dow_arr.astype(np.float32),
            "datetime_ordinal": ordinal_arr.astype(np.float64),
        }
        all_clients.append(client_entry)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n} clients...")

    print(f"  Done. {n} clients processed.")

    result = {
        "clients": all_clients,
        "K": K,
        "tau": tau,
        "config": {k: v for k, v in config.items()
                   if not callable(v) and isinstance(v, (int, float, str, bool, list, tuple, type(None)))},
    }
    return result


def main():
    args = parse_args()
    result = collect_activations(args.checkpoint, args)

    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, f"activations_{args.data_split}.pt")
    torch.save(result, out_path)
    print(f"\nSaved activations to: {out_path}")

    # Print summary
    n_total = sum(len(c["dominant_proto"]) for c in result["clients"])
    print(f"Total windows: {n_total}")
    print(f"Clients: {len(result['clients'])}")
    for c in result["clients"][:3]:
        print(f"  client {c['client_id']}: {len(c['dominant_proto'])} windows, "
              f"dominant_proto range [{c['dominant_proto'].min()}, {c['dominant_proto'].max()}], "
              f"weight shape {c['weights'].shape}")


if __name__ == "__main__":
    main()
