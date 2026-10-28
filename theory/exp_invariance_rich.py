"""
Richer SCM stress-test: does TRAINING + FEDERATING the prototype codebook
de-confound better than a FROZEN RANDOM (but shared) codebook?

Motivation: in the simple linear 2-prototype SCM the trained-federated codebook
and a frozen random shared codebook give identical spurious leakage — the
de-confounding came from the *shared bottleneck*, not from learning Q. Here we
make the invariant mechanism nonlinear & multi-modal (y = sin(2u) + 0.5u, so the
u->y map is non-monotone and needs several prototypes to tile) and widen the
per-client confounding heterogeneity, with K=8 prototypes. If a learned codebook
ever beats random, this is where it should show.

Probe (c_probe=0 independent set):
    leakage   = R^2(s        | z_inv)   spurious info linearly decodable
    retention = R^2(y_clean  | z_inv)   invariant signal f(u) retained

Run:  python theory/exp_invariance_rich.py --device cuda:0 --freeze_prototypes false
      python theory/exp_invariance_rich.py --device cuda:0 --freeze_prototypes true
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/causal_fedts_mpl_{os.getuid()}")

from federated.trainer import build_clients_and_server, federated_train


def f_invariant(u):
    """Nonlinear, non-monotone invariant mechanism f(u)."""
    return np.sin(2.0 * u) + 0.5 * u


def make_client_dict(client_id, c_k, n_train, n_val, scm, rng):
    """X: (N,1,2)=[u,s]; Y:(N,1,1)=y. Constant zero temporal id for STID."""
    def draw(N):
        u = rng.uniform(-3.0, 3.0, N).astype(np.float32)
        y = f_invariant(u) + rng.normal(0.0, scm["se"], N)
        s = c_k * y + rng.normal(0.0, scm["sx"], N)
        X = np.stack([u, s], axis=1)[:, None, :].astype(np.float32)
        Y = y[:, None, None].astype(np.float32)
        return X, Y
    Xtr, Ytr = draw(n_train)
    Xva, Yva = draw(n_val)
    Xte, Yte = draw(n_val)
    zt = lambda N: np.zeros((N, 2), np.float32)
    return {
        "client_id": int(client_id),
        "X_train": Xtr, "Y_train": Ytr, "T_train": zt(n_train),
        "X_val": Xva, "Y_val": Yva, "T_val": zt(n_val),
        "X_test": Xte, "Y_test": Yte, "T_test": zt(n_val),
        "target_mean": None, "target_std": None,
    }


def make_probe(scm, N, rng):
    """Independent probe set: c_probe=0 so s is independent of y/u."""
    u = rng.uniform(-3.0, 3.0, N).astype(np.float32)
    y_clean = f_invariant(u).astype(np.float32)        # invariant target
    s = rng.normal(0.0, scm["sx"], N).astype(np.float32)
    X = np.stack([u, s], axis=1)[:, None, :].astype(np.float32)
    return X, y_clean, s


def r2_linear(features, target):
    A = np.concatenate([features, np.ones((len(features), 1), np.float32)], axis=1)
    beta, *_ = np.linalg.lstsq(A, target, rcond=None)
    pred = A @ beta
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def probe_leakage(clients, server, probe_X, probe_yc, probe_s, device):
    global_protos = server.broadcast()
    x = torch.from_numpy(probe_X).to(device)
    tid = torch.zeros(len(probe_X), 2, device=device)
    leak, ret = [], []
    for c in clients:
        c.set_prototypes(global_protos)
        c.model.eval()
        _, z_inv, _ = c.model(x, temporal_id=tid)
        Z = z_inv.squeeze(1).cpu().numpy()
        leak.append(r2_linear(Z, probe_s))
        ret.append(r2_linear(Z, probe_yc))
    return float(np.mean(leak)), float(np.mean(ret))


def run_one(cbar, spread, scm, args, seed):
    rng = np.random.default_rng(seed)
    # Wide, diverse per-client confounding (not just +/-1): rk spans [-1, 1].
    rk = np.linspace(-1.0, 1.0, args.M)
    cs = cbar + spread * rk
    clients_data = [make_client_dict(k, cs[k], args.N, args.N // 4, scm, rng)
                    for k in range(args.M)]

    model_kwargs = dict(
        input_dim=2, hid_dim=args.hid_dim, seq_len=1, horizon=1, K=args.K, tau=0.1,
        stid_num_layers=2, dropout=0.0,
        use_residual=False, residual_mode="none",
    )
    device = torch.device(args.device)
    clients, server = build_clients_and_server(
        clients_data=clients_data, model_kwargs=model_kwargs,
        batch_size=args.batch_size, lr=args.lr, local_epochs=1, device=device,
        align_mode="none", mi_weight=0.0,
        freeze_prototypes=args.freeze_prototypes,
    )
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
    probe_X, probe_yc, probe_s = make_probe(scm, 4000, np.random.default_rng(9_999))
    return probe_leakage(clients, server, probe_X, probe_yc, probe_s, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--hid_dim", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_rounds", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--freeze_prototypes", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--outdir", type=str, default="theory/out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    scm = dict(se=0.3, sx=1.0)

    cbars = np.round(np.arange(0.0, 1.0001, 0.2), 3)
    rows = []
    print(f"[rich] nonlinear multimodal SCM, K={args.K}, M={args.M}, "
          f"freeze={args.freeze_prototypes}, {args.seeds} seeds")
    print(f"{'cbar':>6} {'leak R2(s|z)':>14} {'retain R2(yc|z)':>16}")
    for cbar in cbars:
        leaks, rets = [], []
        for sd in range(args.seeds):
            lk, rt = run_one(cbar, args.spread, scm, args, seed=sd)
            leaks.append(lk); rets.append(rt)
        leaks, rets = np.array(leaks), np.array(rets)
        rows.append((cbar, leaks.mean(), leaks.std(), rets.mean(), rets.std()))
        print(f"{cbar:6.2f} {leaks.mean():7.4f}+-{leaks.std():.4f} "
              f"{rets.mean():9.4f}+-{rets.std():.4f}")

    csv_path = os.path.join(args.outdir, "rich_curve.csv")
    with open(csv_path, "w") as f:
        f.write("cbar,leak_mean,leak_std,retain_mean,retain_std\n")
        for r in rows:
            f.write("%.4f,%.6f,%.6f,%.6f,%.6f\n" % r)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
