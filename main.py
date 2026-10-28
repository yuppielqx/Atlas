"""
Experiment entry point for CausalFedST.

Usage:
    python main.py

Configuration is set via the config dict in main().
"""

import sys
import os
import json
import argparse
import re
from datetime import datetime
import numpy as np
import torch
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/causal_fedts_matplotlib_{os.getuid()}")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import generate_metrla_clients
from federated.trainer import build_clients_and_server, federated_train
from utils.metrics import compute_all_metrics
from models.encoders import ENCODER_REGISTRY, is_vmap_safe

# CLI dataset name -> on-disk folder under --data_dir. All are BasicTS-format
# [L, N, 3] = (value, time_of_day, day_of_week) with NULL_VAL=0.0; the loader
# reads each dataset's own train/val/test ratios from its desc.json.
DATASET_DIRS = {
    "metr-la": "METR-LA",
    "pems-bay": "PEMS-BAY",
    "pems03": "PEMS03",
    "pems04": "PEMS04",
    "knowair-pm25": "KnowAir-PM25",
    "knowair-full": "KnowAir-Full",
}

# Datasets where input_dim > 1 (multi-feature causal discovery).
MULTIFEATURE_DATASETS = {"knowair-full"}

# Safety valve for residual-pathway extrapolation: clamp predictions in
# normalized (z-score) space to ±SAFE_SIGMA standard deviations before the
# inverse transform. Real targets sit within a few sigma for every dataset
# (METR-LA ~0.8, PEMS-BAY ~2.3, PEMS04 ~4.5, PEMS03 bulk <6), so ±10σ never
# truncates a legitimate prediction yet still catches gross blowups. This is
# dataset-agnostic, unlike the previous raw-unit clip (±100), which was sized
# for METR-LA speed and silently truncated ~60-70% of PEMS03/PEMS04 flow
# targets (values reach 900-1800), inflating their MAE and erasing all
# differences between methods.
SAFE_SIGMA = 10.0


def str2bool(value):
    """Parse explicit boolean CLI values such as true/false or 1/0."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value: true/false")


def parse_args():
    parser = argparse.ArgumentParser(description="CausalFedST Experiment")
    parser.add_argument("--dataset", type=str, default="metr-la",
                        choices=sorted(DATASET_DIRS),
                        help="traffic dataset (each node becomes one client)")
    parser.add_argument("--dataset2", type=str, default=None,
                        choices=sorted(DATASET_DIRS),
                        help="optional second dataset whose nodes are merged into "
                             "the same federated pool, jointly training one shared "
                             "codebook across both datasets")
    parser.add_argument("--data_dir", type=str, default="./datasets",
                        help="root directory for real datasets")
    parser.add_argument("--n_clients", type=int, default=-1,
                        help="number of nodes to use as clients; <=0 means ALL "
                             "nodes (the default). A positive value randomly "
                             "subsamples that many nodes (seeded by --seed).")
    parser.add_argument("--client_ids_file", type=str, default=None,
                        help="optional file with explicit node ids to use as "
                             "clients (comma/space/newline separated). Overrides "
                             "--n_clients for nested subset experiments.")
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--K", type=int, default=10, help="number of global prototypes")
    parser.add_argument("--tau", type=float, default=0.1, help="softmax temperature")
    parser.add_argument("--fedprox_mu", type=float, default=0.0,
                        help="FedProx proximal strength mu (0 disables; use with "
                             "--fl_mode fedavg for the FedProx baseline)")
    parser.add_argument("--ditto_lambda", type=float, default=0.1,
                        help="Ditto personalization strength; use with "
                             "--fl_mode ditto")
    parser.add_argument("--fl_mode", type=str, default="causal_proto",
                        choices=["local", "local_proto", "fedavg", "fedavg_proto",
                                 "fedavg_enc", "fedavg_enc_proto", "causal_proto",
                                 "causal_proto_enc", "ditto"],
                        help="federated aggregation scheme (what is shared each round): "
                             "local (no proto, no share) | local_proto (proto, no share) | "
                             "fedavg (no proto, all params) | fedavg_proto (proto + all params) | "
                             "fedavg_enc (no proto, encoder only) | fedavg_enc_proto (proto, encoder only) | "
                             "causal_proto (default: prototypes + OT/Hungarian) | "
                             "causal_proto_enc (causal_proto + encoder FedAvg)")
    parser.add_argument("--proto_align", type=str, default="hungarian",
                        choices=["none", "ot", "hungarian"],
                        help="align client prototypes to the global anchor before "
                             "aggregation: none, ot (Sinkhorn), or hungarian "
                             "(used by fl_mode=causal_proto/causal_proto_enc)")
    parser.add_argument("--proto_align_eps", type=float, default=0.05,
                        help="Sinkhorn entropic regularization (proto_align=ot)")
    parser.add_argument("--proto_align_iters", type=int, default=50,
                        help="Sinkhorn iteration count (proto_align=ot)")
    parser.add_argument("--mi_weight", type=float, default=0.0,
                        help="Gaussian MI regularisation strength between z_inv "
                             "and z_res (0.0 = disabled, the default; the plain "
                             "forecasting loss attains the best accuracy). Set "
                             ">0 only to reproduce the regulariser ablation.")
    parser.add_argument("--aux_inv_weight", type=float, default=0.0,
                        help="Weight of an auxiliary supervised loss on a z_inv-only "
                             "prediction head (0.0 = disabled). >0 forces the "
                             "invariant predictive signal through the shared codebook "
                             "instead of being bypassed by the local residual z_res.")


    parser.add_argument("--encoder_type", type=str, default="stid",
                        choices=sorted(ENCODER_REGISTRY),
                        help="local client encoder backbone (all output a "
                             "(B,1,hid_dim) window representation). Note: "
                             "timesnet/autoformer/informer run only with "
                             "--vectorized false (auto-enforced)")
    parser.add_argument("--stid_num_layers", type=int, default=3,
                        help="number of encoder blocks: ResMLP blocks (stid) or "
                             "Transformer layers (patchtst); ignored by dlinear")
    parser.add_argument("--encoder_opt", action="append", default=[], metavar="KEY=VAL",
                        help="override an encoder-specific hyperparameter, e.g. "
                             "--encoder_opt patch_len=6 --encoder_opt d_model=128 "
                             "(repeatable; defaults live in models/encoders/<model>.py)")
    parser.add_argument("--graph_context", type=str2bool, default=False,
                        help="append an adjacency-weighted neighbor mean channel "
                             "for graph baselines such as CNFGNN")
    parser.add_argument("--no_temporal_emb", action="store_true", default=False,
                        help="disable time-of-day and day-of-week embedding "
                             "lookups in the STID encoder (pure value-only input)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_residual", type=str2bool, default=True,
                        help="concatenate z_res to predictor input")
    parser.add_argument("--residual_mode", type=str, default="fixed",
                        choices=["none", "fixed", "learnable", "detach", "detach_fixed"],
                        help="residual scaling: none, fixed residual_scale, learnable gate, detach, or detach_fixed")
    parser.add_argument("--residual_scale", type=float, default=1.0,
                        help="fixed residual scale used when residual_mode=fixed or detach_fixed")
    parser.add_argument("--residual_gate_init", type=float, default=-2.0,
                        help="initial logit for residual gate; sigmoid(-2) is about 0.12")
    
    
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="plateau",
                        choices=["none", "plateau"],
                        help="adaptive LR scheduler; plateau reduces LR when val loss stalls")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="multiplicative LR decay factor for plateau scheduler")
    parser.add_argument("--lr_patience", type=int, default=3,
                        help="rounds without val improvement before reducing LR")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="lower bound for adaptive LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 weight decay on local model parameters")
    
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--n_rounds", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8,
                        help="early-stop after this many rounds without val "
                             "improvement (<=0 disables early stopping)")
    
    
    parser.add_argument("--min_delta", type=float, default=0.0,
                        help="minimum val-loss decrease counted as improvement")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_gpus", type=int, default=4,
                        help="number of GPUs for parallel client training")
    
    parser.add_argument("--vectorized", type=str2bool, default=True,
                        help="process each GPU's clients as one vmapped ensemble "
                             "(true within-GPU parallelism; trains all clients "
                             "every round, ignores --fraction)")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--run_name", type=str, default=None,
                        help="name of the run subdirectory under output_dir "
                             "(default: auto timestamp + config)")
    # ── Transfer experiments: load a pretrained codebook and hold it fixed ──
    parser.add_argument("--load_prototypes_from", type=str, default=None,
                        help="path to a checkpoint.pt whose 'server_prototypes' "
                             "codebook Q is loaded before training (transfer)")
    parser.add_argument("--freeze_prototypes", type=str2bool, default=False,
                        help="hold the (loaded) prototype codebook fixed and "
                             "train only the local backbone+head")
    parser.add_argument("--codebook_noise_std", type=float, default=0.0,
                        help="std of Gaussian noise added to each prototype vector "
                             "before client upload (0.0 = disabled)")
    parser.add_argument("--codebook_noise_type", type=str, default="gaussian",
                        choices=["gaussian", "laplace", "uniform", "rademacher"],
                        help="noise distribution type (default: gaussian)")
    parser.add_argument("--codebook_norm", type=str2bool, default=True,
                        help="L2-normalize prototype codebook after every "
                             "optimizer step, aggregation, and init (default: True). "
                             "Set to false to allow prototype norms to drift freely.")
    parser.add_argument("--prototype_init", type=str, default="randn",
                        choices=["randn", "cast"],
                        help="prototype initialisation: randn (default, spread from "
                             "start) or cast (compact uniform near origin, spreads "
                             "during training like CaST)")
    parser.add_argument("--hard_proto", action="store_true",
                        help="use VQ-style hard (one-hot) prototype assignment "
                             "with straight-through gradient estimator instead of "
                             "softmax. Makes each prototype a discrete environment type.")
    # ── Two-stage training (stage 2): freeze the stage-1 invariant path,
    #    train a separate residual branch to restore in-distribution accuracy ──
    parser.add_argument("--separate_residual", type=str2bool, default=False,
                        help="z_res from an independent residual encoder instead "
                             "of the arithmetic complement z - z_inv")
    parser.add_argument("--freeze_encoder", type=str2bool, default=False,
                        help="freeze the invariant path (encoder + codebook); only "
                             "the separate residual branch + predictor train")
    parser.add_argument("--load_model_from", type=str, default=None,
                        help="stage-1 checkpoint: load per-client encoder + codebook "
                             "weights (shape-matched) before stage-2 training")
    # ── Sparse prototype upload: client only uploads its dominant prototype(s) ──
    parser.add_argument("--sparse_proto_upload", type=str2bool, default=False,
                        help="upload only the top-k dominant prototype(s) per client "
                             "instead of all K prototypes (reduces comm. by ~K/topk)")
    parser.add_argument("--proto_topk", type=int, default=1,
                        help="number of top prototypes to upload per client when "
                             "--sparse_proto_upload is True")
    parser.add_argument("--sparse_warmup_rounds", type=int, default=0,
                        help="first N rounds with full prototype upload to allow "
                             "specialisation before switching to sparse mode")
    return parser.parse_args()


def _read_client_ids(path):
    with open(path) as f:
        text = f.read()
    toks = [t for t in re.split(r"[\s,]+", text.strip()) if t]
    if not toks:
        raise ValueError(f"--client_ids_file is empty: {path}")
    try:
        return [int(t) for t in toks]
    except ValueError as exc:
        raise ValueError(
            f"--client_ids_file must contain integer node ids: {path}"
        ) from exc


def _check_transfer_geometry(ckpt, args, src_path):
    src_cfg = ckpt.get("config", {})
    for key in ("K", "hid_dim", "tau", "window_size", "horizon"):
        src, cur = src_cfg.get(key), getattr(args, key)
        if src is not None and src != cur:
            raise ValueError(
                f"Cannot load from {src_path}: source {key}={src} != "
                f"current {key}={cur}. Transfer needs matching latent geometry."
            )
    return src_cfg


def main():
    args = parse_args()
    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    explicit_client_ids = _read_client_ids(args.client_ids_file) if args.client_ids_file else None

    # Each run writes into its own subdirectory under output_dir
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.dataset2:
            ds_tag = f"joint_{args.dataset}_{args.dataset2}"
        else:
            ds_tag = args.dataset
        if explicit_client_ids is not None:
            n_tag = f"ids{len(explicit_client_ids)}"
        else:
            n_tag = "all" if args.n_clients <= 0 else args.n_clients
        run_name = f"{ds_tag}_K{args.K}_n{n_tag}_horizon{args.horizon}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Log the run configuration up front (survives crashes mid-training) ──
    start_time = datetime.now()
    args_record = {
        "run_name": run_name,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join([sys.executable] + sys.argv),
        "args": vars(args),
        "client_ids": explicit_client_ids,
    }
    # Per-run copy alongside the outputs.
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(args_record, f, indent=2)
    # Central append-only log: one line per run for quick grep/diff across runs.
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "runs.log"), "a") as f:
        f.write(json.dumps(args_record) + "\n")

    if args.encoder_type == "cnfgnn":
        args.graph_context = True

    # ── Determine input_dim ──
    is_multifeature = args.dataset in MULTIFEATURE_DATASETS
    if is_multifeature:
        # input_dim is set after loading data (depends on feature count)
        input_dim = None
    else:
        input_dim = 2 if args.graph_context else 1

    use_residual = args.use_residual
    residual_mode = args.residual_mode if use_residual else "none"
    client_label = (
        f"{len(explicit_client_ids)} explicit ids"
        if explicit_client_ids is not None
        else ("ALL" if args.n_clients <= 0 else str(args.n_clients))
    )

    print("=" * 60)
    print("CausalFedST Experiment")
    print(f"  Dataset: {args.dataset} | Clients: {client_label} | Prototypes: {args.K}")
    print(f"  Window: {args.window_size} | Horizon: {args.horizon}")
    print(f"  Input dim: {input_dim if input_dim is not None else 'auto (multi-feature)'}")
    print(f"  Residual: {use_residual} | Residual mode: {residual_mode} | Scale: {args.residual_scale}")
    print(f"  MI weight: {args.mi_weight}" + (" (disabled)" if args.mi_weight == 0.0 else ""))
    print(f"  Proto align: {args.proto_align}"
          + (f" (eps={args.proto_align_eps}, iters={args.proto_align_iters})"
             if args.proto_align == "ot" else ""))
    print(f"  Rounds: {args.n_rounds} | Local epochs: {args.local_epochs}")
    print(f"  LR: {args.lr} | Scheduler: {args.lr_scheduler}"
          + (f" (factor={args.lr_factor}, patience={args.lr_patience}, min_lr={args.min_lr})"
             if args.lr_scheduler == "plateau" else ""))
    print(f"  GPUs: {args.n_gpus}")
    print(f"  Output dir: {run_dir}")
    print("=" * 60)

    # ── Generate data ──

    dataset_dir = DATASET_DIRS[args.dataset]
    n_clients = None if explicit_client_ids is not None else (
        args.n_clients if args.n_clients and args.n_clients > 0 else None
    )

    if args.dataset2:
        # ── Merged multi-dataset: load clients from both datasets ──
        if explicit_client_ids is not None:
            raise ValueError(
                "--client_ids_file is not supported with --dataset2; "
                "all nodes from both datasets are used as clients.")
        dataset2_dir = DATASET_DIRS[args.dataset2]
        print(f"\n[1/4] Loading joint data: {dataset_dir} + {dataset2_dir} "
              f"(input_dim={input_dim})...")
        clients_data1 = generate_metrla_clients(
            n_clients=n_clients,
            client_ids=None,
            window_size=args.window_size,
            horizon=args.horizon,
            use_time_features=False,
            dataset_name=dataset_dir,
            data_dir=args.data_dir,
            seed=args.seed,
            graph_context=args.graph_context,
        )
        clients_data2 = generate_metrla_clients(
            n_clients=n_clients,
            client_ids=None,
            window_size=args.window_size,
            horizon=args.horizon,
            use_time_features=False,
            dataset_name=dataset2_dir,
            data_dir=args.data_dir,
            seed=args.seed,
            graph_context=args.graph_context,
        )
        # Offset client_ids in dataset2 to avoid collisions.
        max_id1 = max(c["client_id"] for c in clients_data1)
        id_offset = max_id1 + 1
        for c in clients_data2:
            c["client_id"] = int(c["client_id"]) + id_offset
            c["dataset_source"] = args.dataset2
        for c in clients_data1:
            c["dataset_source"] = args.dataset
        clients_data = clients_data1 + clients_data2
        print(f"  {args.dataset}: {len(clients_data1)} clients, "
              f"{args.dataset2}: {len(clients_data2)} clients, "
              f"total: {len(clients_data)} clients")
        print(f"  Train samples/client ({args.dataset}): "
              f"{clients_data1[0]['X_train'].shape[0]}")
    else:
        if is_multifeature:
            from data.dataset import generate_multifeature_clients
            print(f"\n[1/4] Loading {dataset_dir} multi-feature data "
                  f"(clients={'explicit ids' if explicit_client_ids is not None else ('ALL nodes' if n_clients is None else n_clients)})...")
            clients_data = generate_multifeature_clients(
                n_clients=n_clients,
                client_ids=explicit_client_ids,
                window_size=args.window_size,
                horizon=args.horizon,
                dataset_name=dataset_dir,
                data_dir=args.data_dir,
                seed=args.seed,
            )
            # Auto-detect input_dim from data
            input_dim = clients_data[0]["X_train"].shape[-1]
            print(f"  Generated {len(clients_data)} clients")
            print(f"  Input dim: {input_dim} features  |  "
                  f"Train samples/client: {clients_data[0]['X_train'].shape[0]}")
        else:
            print(f"\n[1/4] Loading {dataset_dir} data (input_dim={input_dim}, "
                  f"clients={'explicit ids' if explicit_client_ids is not None else ('ALL nodes' if n_clients is None else n_clients)})...")
            clients_data = generate_metrla_clients(
                n_clients=n_clients,
                client_ids=explicit_client_ids,
                window_size=args.window_size,
                horizon=args.horizon,
                use_time_features=False,
                dataset_name=dataset_dir,
                data_dir=args.data_dir,
                seed=args.seed,
                graph_context=args.graph_context,
            )
            print(f"  Generated {len(clients_data)} clients")
            if explicit_client_ids is not None:
                print(f"  Client id file: {args.client_ids_file}")
            print(f"  Train samples/client: {clients_data[0]['X_train'].shape[0]}")

    # ── Build model, clients, server ──
    print("\n[2/4] Building model and federated system...")
    from models.encoders import build_encoder_kwargs
    encoder_kwargs = build_encoder_kwargs(args.encoder_type, args.encoder_opt)
    if args.no_temporal_emb:
        encoder_kwargs["no_temporal_emb"] = True
    # fl_mode determines whether the prototype path is active in the model.
    use_prototype = args.fl_mode not in {"local", "fedavg", "fedavg_enc", "ditto"}
    model_kwargs = dict(
        input_dim=input_dim,
        hid_dim=args.hid_dim,
        seq_len=args.window_size,
        horizon=args.horizon,
        K=args.K,
        tau=args.tau,
        stid_num_layers=args.stid_num_layers,
        dropout=args.dropout,
        use_residual=use_residual,
        residual_mode=residual_mode,
        residual_scale=args.residual_scale,
        residual_gate_init=args.residual_gate_init,
        encoder_type=args.encoder_type,
        encoder_kwargs=encoder_kwargs,
        use_prototype=use_prototype,
        use_aux_inv=(args.aux_inv_weight > 0),
        separate_residual=args.separate_residual,
        normalize_codebook=args.codebook_norm,
        prototype_init=args.prototype_init,
        hard_proto=args.hard_proto,
        multivariate=is_multifeature,
    )

    # ── Transfer / two-stage: optionally load a pretrained codebook Q (held
    #    fixed if frozen). --load_model_from also supplies the codebook. ──
    loaded_prototypes = None
    proto_src = args.load_prototypes_from or args.load_model_from
    if proto_src:
        if not use_prototype:
            raise ValueError(
                f"--load_prototypes_from/--load_model_from requires a "
                f"prototype-using fl_mode; got fl_mode={args.fl_mode}.")
        ckpt = torch.load(proto_src, map_location="cpu",
                          weights_only=False)
        loaded_prototypes = ckpt["server_prototypes"].float()
        src_cfg = _check_transfer_geometry(ckpt, args, proto_src)
        if tuple(loaded_prototypes.shape) != (args.K, args.hid_dim):
            raise ValueError(
                f"Loaded prototypes shape {tuple(loaded_prototypes.shape)} "
                f"!= expected (K={args.K}, hid_dim={args.hid_dim}).")
        print(f"  Loaded prototypes {tuple(loaded_prototypes.shape)} from "
              f"{proto_src} (src dataset="
              f"{src_cfg.get('dataset', '?')}); "
              f"freeze_proto={args.freeze_prototypes} freeze_enc={args.freeze_encoder}")

    clients, server = build_clients_and_server(
        clients_data=clients_data,
        model_kwargs=model_kwargs,
        batch_size=args.batch_size,
        lr=args.lr,
        local_epochs=args.local_epochs,
        device=device,
        weight_decay=args.weight_decay,
        align_mode=args.proto_align,
        align_eps=args.proto_align_eps,
        align_iters=args.proto_align_iters,
        mi_weight=args.mi_weight,
        freeze_prototypes=args.freeze_prototypes,
        initial_prototypes=loaded_prototypes,
        inv_weight=args.aux_inv_weight,
        freeze_encoder=args.freeze_encoder,
        fedprox_mu=args.fedprox_mu,
        ditto_lambda=args.ditto_lambda,
        normalize_codebook=args.codebook_norm,
        codebook_noise_std=args.codebook_noise_std,
        codebook_noise_type=args.codebook_noise_type,
    )

    # Two-stage: load the stage-1 per-client invariant path (encoder + codebook).
    # Shape-matched keys only, so the fresh stage-2 modules (separate residual
    # encoder, wider predictor) keep their initialisation.
    if args.load_model_from:
        ck = torch.load(args.load_model_from, map_location="cpu", weights_only=False)
        cms = ck["client_models"]
        if len(cms) != len(clients):
            raise ValueError(
                f"--load_model_from has {len(cms)} client models != "
                f"{len(clients)} current clients (match --n_clients/--seed/--dataset).")
        n_loaded = 0
        for cl, sd in zip(clients, cms):
            msd = cl.model.state_dict()
            filtered = {k: v for k, v in sd.items()
                        if k in msd and tuple(v.shape) == tuple(msd[k].shape)}
            cl.model.load_state_dict(filtered, strict=False)
            n_loaded = len(filtered)
        print(f"  Loaded stage-1 per-client weights: {n_loaded} shape-matched "
              f"keys/client from {args.load_model_from}")

    # Add test loaders for final evaluation
    from torch.utils.data import DataLoader, TensorDataset
    for i, client in enumerate(clients):
        test_tensors = [
            torch.from_numpy(clients_data[i]["X_test"]),
            torch.from_numpy(clients_data[i]["Y_test"]),
        ]
        if "T_test" in clients_data[i]:
            test_tensors.append(torch.from_numpy(clients_data[i]["T_test"]))
        client.test_loader = DataLoader(
            TensorDataset(*test_tensors),
            batch_size=args.batch_size,
            shuffle=False,
        )

    total_params = sum(p.numel() for p in clients[0].model.parameters())
    # In bypass modes the prototype params are inert; report the effective count.
    eff_params = sum(p.numel() for n, p in clients[0].model.named_parameters()
                     if use_prototype or "prototype" not in n)
    print(f"  Model params/client: {total_params:,}" +
          ("" if use_prototype else f" (effective {eff_params:,}, prototype inert)"))
    print(f"  fl_mode: {args.fl_mode} | use_prototype: {use_prototype} | Residual: {clients[0].model.use_residual}")

    # ── Federated training ──
    # timesnet/autoformer/informer keep their FFT/topk/ProbSparse mechanisms,
    # which break torch.func.vmap, so force the per-client path for them.
    vectorized = args.vectorized
    if vectorized and not is_vmap_safe(args.encoder_type):
        print(f"  [warn] encoder_type={args.encoder_type} is not vmap-safe; "
              f"falling back to --vectorized false (per-client training). "
              f"Tip: use --n_gpus 1 for these encoders.")
        vectorized = False
    # Vectorized parameter-FedAvg is implemented for a single ensemble group only;
    # with >1 GPU group, fall back to the per-client path.
    fedavg_modes = {"fedavg", "fedavg_proto", "fedavg_enc", "fedavg_enc_proto",
                    "causal_proto_enc"}
    if vectorized and args.fl_mode == "ditto":
        print("  [warn] fl_mode=ditto uses the per-client personalized path; "
              "falling back to --vectorized false.")
        vectorized = False
    if vectorized and args.fl_mode in fedavg_modes and args.n_gpus > 1:
        print(f"  [warn] fl_mode={args.fl_mode} param-FedAvg needs a single ensemble "
              f"group; falling back to --vectorized false (use --n_gpus 1 to keep "
              f"the fast path).")
        vectorized = False
    # Joint multi-dataset: clients from different datasets have different sample
    # counts (different time-series length), so they can't be stacked in one
    # vmapped ensemble. Fall back to the per-client path.
    if vectorized and args.dataset2:
        print("  [warn] --dataset2 merges datasets with different sample counts; "
              "falling back to --vectorized false (per-client training).")
        vectorized = False

    # ── Sparse upload: requires the per-client path (vmap doesn't expose weights).
    if args.sparse_proto_upload and vectorized:
        print("  [warn] --sparse_proto_upload requires the per-client path; "
              "falling back to --vectorized false.")
        vectorized = False

    print(f"\n[3/4] Federated training ({args.n_rounds} rounds)...")
    history = federated_train(
        clients=clients,
        server=server,
        n_rounds=args.n_rounds,
        fraction=args.fraction,
        n_gpus=args.n_gpus,
        device=device,
        verbose=True,
        patience=args.patience if args.patience and args.patience > 0 else None,
        min_delta=args.min_delta,
        vectorized=vectorized,
        fl_mode=args.fl_mode,
        metrics_csv_path=os.path.join(run_dir, "metrics.csv"),
        lr_scheduler=args.lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
        freeze_prototypes=args.freeze_prototypes,
        freeze_encoder=args.freeze_encoder,
        sparse_proto_upload=args.sparse_proto_upload,
        proto_topk=args.proto_topk,
        sparse_warmup_rounds=args.sparse_warmup_rounds,
        normalize_codebook=args.codebook_norm,
        codebook_noise_std=args.codebook_noise_std,
        codebook_noise_type=args.codebook_noise_type,
    )

    # ── Evaluation (per-client inverse normalization, following BasicTS convention) ──
    print("\n[4/4] Final evaluation...")
    global_protos = server.broadcast()
    all_y_true, all_y_pred = [], []
    # z_inv-only readout (the de-confounded predictor), reported when the aux head
    # is active so we can see how good the invariant channel is on its own.
    eval_inv = bool(getattr(clients[0].model, "use_aux_inv", False))
    all_y_pred_inv = []

    for i, client in enumerate(clients):
        client.set_prototypes(global_protos)
        client.model.eval()

        target_mean = clients_data[i]["target_mean"]
        target_std = clients_data[i]["target_std"]

        client_y_true, client_y_pred, client_y_pred_inv = [], [], []
        with torch.no_grad():
            for batch in client.test_loader:
                batch_x, batch_y = batch[:2]
                batch_t = batch[2] if len(batch) > 2 else None
                batch_x = batch_x.to(device)
                if batch_t is not None:
                    batch_t = batch_t.to(device)
                if eval_inv:
                    y_hat, y_hat_inv, _, _ = client.model(
                        batch_x, temporal_id=batch_t, return_inv=True)
                    client_y_pred_inv.append(y_hat_inv.cpu().numpy())
                else:
                    y_hat, _, _ = client.model(batch_x, temporal_id=batch_t)
                client_y_true.append(batch_y[..., :1].cpu().numpy())
                client_y_pred.append(y_hat.cpu().numpy())

        yt = np.concatenate(client_y_true, axis=0)
        yp = np.concatenate(client_y_pred, axis=0)

        # Safety valve in normalized space (see SAFE_SIGMA): clamp before the
        # inverse transform so the bound scales with each client's own std.
        yp = np.clip(yp, -SAFE_SIGMA, SAFE_SIGMA)

        # Inverse normalization: normalized → original scale
        yt_raw = yt * target_std + target_mean
        yp_raw = yp * target_std + target_mean
        all_y_true.append(yt_raw)
        all_y_pred.append(yp_raw)
        if eval_inv:
            yp_i = np.clip(np.concatenate(client_y_pred_inv, axis=0),
                           -SAFE_SIGMA, SAFE_SIGMA)
            all_y_pred_inv.append(yp_i * target_std + target_mean)

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)

    # All four datasets use NULL_VAL=0.0 (missing/invalid encoded as 0), matching
    # BasicTS' masked metrics, so exclude near-zero targets everywhere.
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    mask = finite_mask & (y_true > 0.1)

    metrics = compute_all_metrics(y_true, y_pred, mask=mask)

    # z_inv-only readout metrics (the de-confounded predictor), if the aux head ran.
    if eval_inv and all_y_pred_inv:
        y_pred_inv = np.concatenate(all_y_pred_inv, axis=0)
        mask_inv = np.isfinite(y_true) & np.isfinite(y_pred_inv) & (y_true > 0.1)
        inv_metrics = compute_all_metrics(y_true, y_pred_inv, mask=mask_inv)
        for k, v in inv_metrics.items():
            metrics[f"zinv_{k}"] = float(v)

    # ── Per-dataset metrics (when multiple datasets are merged) ──
    if args.dataset2:
        from collections import defaultdict
        ds_yt = defaultdict(list)
        ds_yp = defaultdict(list)
        for i, (yt_raw, yp_raw) in enumerate(zip(all_y_true, all_y_pred)):
            ds = clients_data[i].get("dataset_source", args.dataset)
            ds_yt[ds].append(yt_raw)
            ds_yp[ds].append(yp_raw)
        for ds_name in sorted(ds_yt.keys()):
            ds_y_true = np.concatenate(ds_yt[ds_name], axis=0)
            ds_y_pred = np.concatenate(ds_yp[ds_name], axis=0)
            ds_mask = (np.isfinite(ds_y_true) & np.isfinite(ds_y_pred)
                       & (ds_y_true > 0.1))
            ds_metrics = compute_all_metrics(ds_y_true, ds_y_pred, mask=ds_mask)
            for k, v in ds_metrics.items():
                metrics[f"{ds_name}_{k}"] = float(v)

    # ── Per-client metrics (macro-averaged) ──
    per_client_mae = []
    per_client_records = []
    nonfinite_clients = []
    for i, (yt_raw, yp_raw) in enumerate(zip(all_y_true, all_y_pred)):
        # yp_raw was already clamped in normalized space before inversion.
        finite_i = np.isfinite(yt_raw) & np.isfinite(yp_raw)
        if not finite_i.all():
            nonfinite_clients.append({
                "client_id": int(clients_data[i].get("client_id", i)),
                "nonfinite_count": int((~finite_i).sum()),
                "target_std": float(clients_data[i]["target_std"]),
            })
        m = finite_i & (yt_raw > 0.1)
        if m.sum() > 0:
            mae_i = float(np.abs(yt_raw[m] - yp_raw[m]).mean())
            per_client_mae.append(mae_i)
            per_client_records.append({
                "client_id": int(clients_data[i].get("client_id", i)),
                "raw_mae": mae_i,
                "target_std": float(clients_data[i]["target_std"]),
            })

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    if args.dataset2:
        # Per-dataset summary line for quick comparison.
        print("  ---")
        for ds_name in sorted(set(
            c.get("dataset_source", args.dataset) for c in clients_data
        )):
            ds_mae = metrics.get(f"{ds_name}_MAE", float("nan"))
            ds_rmse = metrics.get(f"{ds_name}_RMSE", float("nan"))
            print(f"  [{ds_name}] MAE={ds_mae:.4f}  RMSE={ds_rmse:.4f}")
    if per_client_mae:
        pca = np.array(per_client_mae)
        print(f"  Per-client MAE (macro): mean {pca.mean():.4f}, "
              f"median {np.median(pca):.4f}, "
              f"min {pca.min():.4f}, max {pca.max():.4f}")
        metrics["per_client_MAE_mean"] = float(pca.mean())
        metrics["per_client_MAE_median"] = float(np.median(pca))
        worst_clients = sorted(per_client_records, key=lambda r: r["raw_mae"], reverse=True)[:10]
        print("  Worst clients (raw MAE):")
        for r in worst_clients:
            print(f"    id={r['client_id']} mae={r['raw_mae']:.4f} target_std={r['target_std']:.4f}")
        if nonfinite_clients:
            print("  Non-finite prediction clients:")
            for r in nonfinite_clients[:10]:
                print(f"    id={r['client_id']} nonfinite={r['nonfinite_count']} target_std={r['target_std']:.4f}")
    else:
        worst_clients = []

    best_round = history.get("best_round", len(history["val_loss"]) - 1)
    print(f"  Best round: {best_round} (val loss {history.get('best_val_loss', float('nan')):.4f})")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")

    # ── Save results ──
    results = {
        "config": vars(args),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "best_round": int(history.get("best_round", -1)),
        "best_val_loss": float(history.get("best_val_loss", float("nan"))),
        "worst_clients": worst_clients,
        "nonfinite_clients": nonfinite_clients,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Save checkpoint ──
    # The checkpoint stores the deployable best-round state: personalized client
    # weights paired with one shared best-round global prototype. Make the
    # client state_dicts explicit about using that global prototype too.
    checkpoint_global_protos = server.broadcast()
    for client in clients:
        client.set_prototypes(checkpoint_global_protos)
    checkpoint = {
        "checkpoint_semantics": "best_personalized_client_models_with_shared_global_prototype",
        "server_prototypes": checkpoint_global_protos.cpu(),
        "client_models": [client.model.state_dict() for client in clients],
        "clients_data_meta": [
            {"client_id": d.get("client_id", i),
             "target_mean": d["target_mean"],
             "target_std": d["target_std"]}
            for i, d in enumerate(clients_data)
        ],
        "history": history,
        "proto_history": history.get("proto_history", []),
        "proto_history_rounds": history.get("proto_history_rounds", []),
        "config": vars(args),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    torch.save(checkpoint, os.path.join(run_dir, "checkpoint.pt"))
    print(f"  Checkpoint saved to {run_dir}/checkpoint.pt")

    # ── Plot training curves ──
    fig, ax = plt.subplots(figsize=(8, 4))
    rounds = list(range(len(history["train_loss"])))
    ax.plot(rounds, history["train_loss"], label="Train Loss", linewidth=1.5)
    ax.plot(rounds, history["val_loss"], label="Val Loss (mean)", linewidth=1.5)
    if history.get("val_min") and history.get("val_max"):
        ax.fill_between(rounds, history["val_min"], history["val_max"],
                        alpha=0.15, color="C1", label="Val min–max (per client)")
    if history.get("best_round", -1) >= 0:
        ax.axvline(history["best_round"], color="gray", linestyle="--",
                   linewidth=1.0, label=f"Best (round {history['best_round']})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Masked MAE Loss")
    ax.set_title(f"CausalFedST Training (K={args.K}, clients={args.n_clients})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curve.png"), dpi=150)
    print(f"\n  Results saved to {run_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
