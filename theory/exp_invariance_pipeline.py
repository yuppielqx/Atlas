"""
Pipeline validation of Proposition 1'(iii) through the *actual* CausalFedST
stack (encoder -> K=1 prototype matching -> predictor, FedAvg over prototypes).

Shows the same effect as the standalone surrogate, but inside the real model:
the spurious leakage of the shared representation z_inv, measured by linear
probing, grows with the federation-averaged spurious correlation cbar and
vanishes at cbar = 0.

The linear-Gaussian SCM is embedded into the forecasting interface as a
degenerate one-step problem:  window=1, input_dim=2 (channels = [u, s]),
horizon=1, target = y. Masking is disabled (target_mean/std = None).

Leakage metric (probing): freeze the trained global model; on an INDEPENDENT
probe set with c_probe = 0 (so s is independent of y and u), regress the probe's
u and s on z_inv and report
    leakage   = R^2(s | z_inv)     (spurious info linearly decodable from shared rep)
    retention = R^2(u | z_inv)     (invariant info retained)

Run (uses one GPU; avoids the unstable multi-GPU path):
    python theory/exp_invariance_pipeline.py --device cuda:0
Outputs:
    theory/out/pipeline_curve.csv
    theory/out/fig_invariance_pipeline.png
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/causal_fedts_mpl_{os.getuid()}")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from federated.trainer import build_clients_and_server, federated_train


# ----------------------------------------------------------------------------- #
#  Synthetic SCM -> clients_data dicts (forecasting interface)                   #
# ----------------------------------------------------------------------------- #
def make_client_dict(client_id, c_k, n_train, n_val, scm, rng):
    """One client's data dict in the schema generate_metrla_clients produces.

    X: (N, 1, 2) = [u, s] at a single timestep; Y: (N, 1, 1) = y; no temporal ids.
    target_mean/std are None so the masked loss keeps every sample.
    """
    def draw(N):
        u = rng.normal(0.0, scm["su"], N)
        y = scm["alpha"] * u + rng.normal(0.0, scm["se"], N)
        s = c_k * y + rng.normal(0.0, scm["sx"], N)
        X = np.stack([u, s], axis=1)[:, None, :].astype(np.float32)   # (N,1,2)
        Y = y[:, None, None].astype(np.float32)                       # (N,1,1)
        return X, Y
    Xtr, Ytr = draw(n_train)
    Xva, Yva = draw(n_val)
    Xte, Yte = draw(n_val)
    # STID's encoder requires a temporal id; the SCM has no time, so feed a
    # constant zero (time_of_day=day_of_week=0) — a harmless constant embedding
    # that leaves the [u, s] series path untouched.
    zt = lambda N: np.zeros((N, 2), np.float32)
    return {
        "client_id": int(client_id),
        "X_train": Xtr, "Y_train": Ytr, "T_train": zt(n_train),
        "X_val": Xva, "Y_val": Yva, "T_val": zt(n_val),
        "X_test": Xte, "Y_test": Yte, "T_test": zt(n_val),
        "target_mean": None, "target_std": None,
    }


def make_probe(scm, N, rng):
    """Independent probe set with c_probe = 0 (s independent of y, u)."""
    u = rng.normal(0.0, scm["su"], N)
    s = rng.normal(0.0, scm["sx"], N)          # c_probe = 0 -> s independent of u, y
    X = np.stack([u, s], axis=1)[:, None, :].astype(np.float32)
    return X, u.astype(np.float32), s.astype(np.float32)


# ----------------------------------------------------------------------------- #
#  Probing                                                                       #
# ----------------------------------------------------------------------------- #
def r2_linear(features, target):
    """R^2 of OLS predicting target from features (+ intercept)."""
    A = np.concatenate([features, np.ones((len(features), 1), np.float32)], axis=1)
    beta, *_ = np.linalg.lstsq(A, target, rcond=None)
    pred = A @ beta
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def probe_leakage(clients, server, probe_X, probe_u, probe_s, device):
    """Average over clients of R^2(s|z_inv) and R^2(u|z_inv) on the probe set."""
    global_protos = server.broadcast()
    x = torch.from_numpy(probe_X).to(device)
    tid = torch.zeros(len(probe_X), 2, device=device)   # constant temporal id (see make_client_dict)
    leak, ret = [], []
    for c in clients:
        c.set_prototypes(global_protos)
        c.model.eval()
        _, z_inv, _ = c.model(x, temporal_id=tid)       # z_inv: (B,1,hid)
        Z = z_inv.squeeze(1).cpu().numpy()
        leak.append(r2_linear(Z, probe_s))
        ret.append(r2_linear(Z, probe_u))
    return float(np.mean(leak)), float(np.mean(ret))


# ----------------------------------------------------------------------------- #
#  One federated run at a given cbar                                             #
# ----------------------------------------------------------------------------- #
def run_one(cbar, spread, scm, args, seed):
    rng = np.random.default_rng(seed)
    rk = np.where(np.arange(args.M) < args.M // 2, 1.0, -1.0)
    cs = cbar + spread * rk
    clients_data = [make_client_dict(k, cs[k], args.N, args.N // 4, scm, rng)
                    for k in range(args.M)]

    model_kwargs = dict(
        input_dim=2, hid_dim=args.hid_dim, seq_len=1, horizon=1, K=args.K, tau=0.1,
        stid_num_layers=2, dropout=0.0,
        use_residual=args.use_residual, residual_mode=args.residual_mode,
    )
    device = torch.device(args.device)
    clients, server = build_clients_and_server(
        clients_data=clients_data, model_kwargs=model_kwargs,
        batch_size=args.batch_size, lr=args.lr, local_epochs=1, device=device,
        align_mode="none", mi_weight=0.0,
        freeze_prototypes=args.freeze_prototypes,
    )
    # Independent prototype init per seed (server holds the shared anchor; FedAvg
    # syncs clients each round, so randomness lives in the server init seed).
    torch.manual_seed(10_000 + seed)
    import torch.nn.functional as F
    server.global_prototypes = F.normalize(
        torch.randn(args.K, args.hid_dim, device=device), dim=-1, eps=1e-6)

    federated_train(
        clients=clients, server=server, n_rounds=args.n_rounds, fraction=1.0,
        n_gpus=1, device=device, verbose=False, patience=None,
        vectorized=False, lr_scheduler="none",
        freeze_prototypes=args.freeze_prototypes,
    )
    probe_X, probe_u, probe_s = make_probe(scm, 4000, np.random.default_rng(9_999))
    return probe_leakage(clients, server, probe_X, probe_u, probe_s, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=30)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--hid_dim", type=int, default=16)
    ap.add_argument("--K", type=int, default=2,
                    help="prototypes; K>=2 (K=1 makes z_inv constant: softmax over 1)")
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_rounds", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--use_residual", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--residual_mode", type=str, default="none")
    ap.add_argument("--freeze_prototypes", type=lambda s: s.lower() == "true", default=False,
                    help="hold the (random, shared) codebook fixed: no local "
                         "training, no FedAvg. Compares a frozen random shared "
                         "codebook against the trained+federated one.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--outdir", type=str, default="theory/out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    scm = dict(alpha=1.0, su=1.0, se=0.5, sx=1.0)

    cbars = np.round(np.arange(0.0, 1.0001, 0.2), 3)
    rows = []
    print(f"[pipeline] real CausalFedST, K={args.K}, use_residual={args.use_residual}, "
          f"M={args.M}, {args.seeds} seeds")
    print(f"{'cbar':>6} {'leak R2(s|z)':>14} {'retain R2(u|z)':>15}")
    for cbar in cbars:
        leaks, rets = [], []
        for sd in range(args.seeds):
            lk, rt = run_one(cbar, args.spread, scm, args, seed=sd)
            leaks.append(lk); rets.append(rt)
        leaks, rets = np.array(leaks), np.array(rets)
        rows.append((cbar, leaks.mean(), leaks.std(), rets.mean(), rets.std()))
        print(f"{cbar:6.2f} {leaks.mean():7.4f}+-{leaks.std():.4f} "
              f"{rets.mean():9.4f}+-{rets.std():.4f}")

    csv_path = os.path.join(args.outdir, "pipeline_curve.csv")
    with open(csv_path, "w") as f:
        f.write("cbar,leak_mean,leak_std,retain_mean,retain_std\n")
        for r in rows:
            f.write("%.4f,%.6f,%.6f,%.6f,%.6f\n" % r)
    print(f"Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = [r[0] for r in rows]
    ax.errorbar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], fmt="o-",
                capsize=3, color="C3", label=r"spurious leakage $R^2(s\,|\,z_{inv})$")
    ax.errorbar(x, [r[3] for r in rows], yerr=[r[4] for r in rows], fmt="s-",
                capsize=3, color="C0", label=r"invariant retention $R^2(u\,|\,z_{inv})$")
    ax.set_xlabel(r"federation-averaged spurious correlation  $\bar{c}$")
    ax.set_ylabel(r"probed $R^2$ of shared representation $z_{inv}$")
    ax.set_title(f"Real CausalFedST (K={args.K}): spurious leakage grows with "
                 r"$\bar{c}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig_path = os.path.join(args.outdir, "fig_invariance_pipeline.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
