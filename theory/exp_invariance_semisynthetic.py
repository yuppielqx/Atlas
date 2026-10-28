"""
Semisynthetic validation of Proposition 1'(iii) using REAL traffic data.

Same setup as exp_invariance_standalone.py — a single shared unit direction p
trained by spherical FedAvg with personalized linear heads — but the invariant
signal u is real METR-LA traffic speed instead of synthetic Gaussian draws.

Data-generating process per client k:
    u = real METR-LA speed(t)                                    (invariant cause)
    eps ~ N(0, se^2);  y = alpha*u + eps                         (target)
    xi ~ N(0, sx^2);   s = c_k * y + xi                          (spurious)
    z = [u, s]                                                    (observed)

Model per client k:  y_hat = a_k * (z @ p) + b_k
p is shared (FedAvg + renormalisation); (a_k, b_k) are personalised.

Leakage metric:  |p_s| / ||p||  (spurious component of the shared direction)

Comparison:  Federated (FedAvg p)  vs  Per-client (non-federated OLS)

Run:
    python theory/exp_invariance_semisynthetic.py
Outputs:
    theory/out/semisynthetic_curve.csv
    theory/out/fig_invariance_semisynthetic.png
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

from data.dataset import load_metrla_raw


# --------------------------------------------------------------------------- #
#  SCM moments / closed-form optimum (IDENTICAL to standalone)                #
# --------------------------------------------------------------------------- #

class SCM:
    """Population/sample moments for the linear-Gaussian SCM.

    After z-scoring real METR-LA speeds, Var(u) = 1.0 = su², matching
    standalone's su=1.0 scaling.  The closed-form formulae are identical
    to standalone.SCM.
    """

    def __init__(self, alpha=1.0, su=1.0, se=0.5, sx=1.0):
        self.alpha, self.su2, self.se2, self.sx2 = alpha, su**2, se**2, sx**2
        self.VY = alpha**2 * su**2 + se**2

    def rho2(self, psi, c):
        """Squared correlation between y and the projected feature t = z . (cos,sin)."""
        pu, ps = np.cos(psi), np.sin(psi)
        cov = pu * self.alpha * self.su2 + ps * c * self.VY
        var = (pu**2 * self.su2 + 2 * pu * ps * c * self.alpha * self.su2
               + ps**2 * (c**2 * self.VY + self.sx2))
        return cov**2 / (self.VY * var)

    def closed_form_leakage(self, cs, ws, grid=4001):
        """Leakage |sin psi*| of the federated optimum argmax_psi sum_k w_k rho2_k."""
        psis = np.linspace(-np.pi / 2, np.pi / 2, grid)
        G = np.array([sum(w * self.rho2(p, c) for w, c in zip(ws, cs)) for p in psis])
        return abs(np.sin(psis[G.argmax()]))


# --------------------------------------------------------------------------- #
#  Semisynthetic data generation (replaces SCM.sample)                        #
# --------------------------------------------------------------------------- #

def sample_client(values_norm, node_id, c_k, alpha, se, sx, N, rng):
    """Draw N samples for a client with spurious strength c_k.

    Mirrors SCM.sample() in standalone, but u is z-scored real METR-LA speed
    (so Var(u) ≈ 1.0, matching standalone's su=1.0).
    Returns z(N,2) = [u, s], y(N,) where y = alpha*u + eps.

    Args:
        values_norm: (L, num_nodes) float32 — z-scored real traffic speeds
        node_id: which sensor to use
        c_k: spurious strength for this client
        alpha: invariant coefficient (y = alpha*u + eps)
        se: std of target noise eps
        sx: std of spurious noise xi
        N: number of samples
        rng: numpy random generator

    Returns:
        z: (N, 2) float32  [u, s]
        y: (N,)  float32   target
    """
    L = values_norm.shape[0]
    train_end = int(L * 0.7)
    idx = rng.choice(train_end, size=N, replace=True)
    u = values_norm[idx, int(node_id)].astype(np.float32)

    # Target: y = alpha*u + eps  (matches standalone SCM)
    eps = rng.normal(0.0, se, N).astype(np.float32)
    y = alpha * u + eps

    # Spurious feature: s = c_k * y + xi
    xi = rng.normal(0.0, sx, N).astype(np.float32)
    s = c_k * y + xi

    z = np.stack([u, s], axis=1).astype(np.float32)
    return z, y


# --------------------------------------------------------------------------- #
#  Federated training (IDENTICAL to standalone)                               #
# --------------------------------------------------------------------------- #

def leakage_of(p):
    """Spurious component of a 2-vector direction: |p_s| / ||p||  in [0,1]."""
    p = np.asarray(p, dtype=float)
    return abs(p[1]) / (np.linalg.norm(p) + 1e-12)


def _fedavg_once(client_data, Zs, Ys, n_rounds, local_steps, lr, seed):
    """One FedAvg run from a random init. Returns (p unit np(2,), final train MSE)."""
    g = torch.Generator().manual_seed(seed)
    p_global = torch.randn(2, generator=g)
    p_global = p_global / p_global.norm()
    M = len(client_data)
    heads = [{"a": torch.tensor(1.0, requires_grad=True),
              "b": torch.tensor(0.0, requires_grad=True)} for _ in range(M)]
    final_losses = [0.0] * M
    for _ in range(n_rounds):
        p_locals = []
        for k in range(M):
            p_loc = p_global.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([p_loc, heads[k]["a"], heads[k]["b"]], lr=lr)
            loss = None
            for _ in range(local_steps):
                opt.zero_grad()
                pred = heads[k]["a"] * (Zs[k] @ p_loc) + heads[k]["b"]
                loss = ((pred - Ys[k]) ** 2).mean()
                loss.backward()
                opt.step()
            p_locals.append(p_loc.detach())
            final_losses[k] = float(loss.detach())
        p_global = torch.stack(p_locals).mean(0)
        p_global = p_global / (p_global.norm() + 1e-12)
    return p_global.numpy(), float(np.mean(final_losses))


def fedavg_train(client_data, n_rounds=80, local_steps=2, lr=0.05, seed=0, n_restarts=4):
    """FedAvg over a shared unit direction p with personalised linear heads.

    Identical to standalone.fedavg_train.
    """
    Zs = [torch.from_numpy(z) for z, _ in client_data]
    Ys = [torch.from_numpy(y) for _, y in client_data]
    best_p, best_loss = None, float("inf")
    for r in range(n_restarts):
        p, loss = _fedavg_once(client_data, Zs, Ys, n_rounds, local_steps, lr,
                               seed=seed * 97 + r)
        if loss < best_loss:
            best_loss, best_p = loss, p
    return best_p


def per_client_leakage(client_data):
    """Average leakage of per-client ERM (each client fits its own direction)."""
    leaks = []
    for z, y in client_data:
        A = np.concatenate([z, np.ones((len(z), 1), np.float32)], axis=1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        leaks.append(leakage_of(beta[:2]))
    return float(np.mean(leaks))


# --------------------------------------------------------------------------- #
#  Experiment driver                                                           #
# --------------------------------------------------------------------------- #

def build_clients(values, cbar, spread, M, N, alpha, se, sx, seed):
    """Build client data using real METR-LA speeds.

    c_k = cbar + spread * r_k  with r_k a fixed zero-mean +/-1 pattern.
    """
    rng = np.random.default_rng(seed)
    num_nodes = values.shape[1]
    node_rng = np.random.default_rng(seed * 3)
    node_ids = node_rng.choice(num_nodes, M, replace=False)
    node_ids = np.sort(node_ids)

    rk = np.where(np.arange(M) < M // 2, 1.0, -1.0)
    cs = cbar + spread * rk

    data = [sample_client(values, nid, c, alpha, se, sx, N, rng)
            for nid, c in zip(node_ids, cs)]
    ws = np.full(M, 1.0 / M)
    return data, cs, ws


def main():
    ap = argparse.ArgumentParser(
        description="Semisynthetic: real METR-LA data + standalone linear FedAvg")
    ap.add_argument("--M", type=int, default=50,
                    help="number of clients (sensors)")
    ap.add_argument("--N", type=int, default=2000,
                    help="samples per client")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="invariant coefficient (y = alpha*u + eps)")
    ap.add_argument("--se", type=float, default=0.5,
                    help="target noise std")
    ap.add_argument("--sx", type=float, default=2.0,
                    help="spurious noise std; larger keeps invariant direction "
                         "globally dominant so FedAvg reliably finds it")
    ap.add_argument("--spread", type=float, default=0.5,
                    help="fixed spread for panel (a)")
    ap.add_argument("--seeds", type=int, default=5,
                    help="independent restarts per condition")
    ap.add_argument("--n_rounds", type=int, default=60)
    ap.add_argument("--data_dir", type=str, default="datasets")
    ap.add_argument("--outdir", type=str, default="theory/out")
    ap.add_argument("--load", action="store_true",
                    help="skip computation; load rows from semisynthetic_rows.npz and plot only")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    npz_path = os.path.join(args.outdir, "semisynthetic_rows.npz")

    if args.load:
        # ---- Load pre-computed rows and plot directly ----
        print(f"Loading pre-computed rows from {npz_path}")
        loaded = np.load(npz_path, allow_pickle=True)
        rows = [tuple(r) for r in loaded["rows"]]
        print(f"Loaded {len(rows)} rows")
    else:
        # ---- Full computation ----
        # Load real METR-LA data once and z-score normalise so Var(u) ≈ 1.0
        values_raw, _ = load_metrla_raw("METR-LA", args.data_dir)
        train_end = int(values_raw.shape[0] * 0.7)
        u_mean = float(np.mean(values_raw[:train_end]))
        u_std  = float(np.std(values_raw[:train_end]))
        values = (values_raw - u_mean) / u_std
        Vy = args.alpha**2 * 1.0 + args.se**2
        print(f"Loaded METR-LA: {values.shape[0]} timesteps, {values.shape[1]} nodes")
        print(f"z-scored u: mean={u_mean:.2f}, std={u_std:.2f}")
        print(f"SCM: alpha={args.alpha}, se={args.se}, sx={args.sx}, VY={Vy:.3f}")

        scm = SCM(alpha=args.alpha, su=1.0, se=args.se, sx=args.sx)
        cbars = np.round(np.arange(0.0, 1.0001, 0.1), 3)
        rows = []

        # ---- Panel (a): leakage vs cbar, spread fixed ----
        print("\n[Panel a] leakage vs cbar (spread=%.2f)" % args.spread)
        print(f"{'cbar':>6} {'fed':>14} {'closed-form':>12} {'per-client':>11}")
        for cbar in cbars:
            fed, pcl = [], []
            closed = None
            for sd in range(args.seeds):
                data, cs, ws = build_clients(values, cbar, args.spread, args.M, args.N,
                                             args.alpha, args.se, args.sx, seed=1000 + sd)
                p_hat = fedavg_train(data, n_rounds=args.n_rounds, seed=sd)
                fed.append(leakage_of(p_hat))
                pcl.append(per_client_leakage(data))
                if closed is None:
                    closed = scm.closed_form_leakage(cs, ws)
            fed = np.array(fed)
            rows.append(("a", cbar, fed.mean(), fed.std(), closed, float(np.mean(pcl))))
            print(f"{cbar:6.2f} {fed.mean():7.4f}+-{fed.std():.4f} {closed:12.4f} {np.mean(pcl):11.4f}")

        # ---- Panel (b): leakage vs spread, cbar fixed at 0 ----
        spreads = [0.3, 0.6, 0.9, 1.2, 1.5]
        print("\n[Panel b] control: leakage vs spread (cbar=0) -> should stay ~0")
        print(f"{'spread':>6} {'fed':>14} {'closed-form':>12} {'per-client':>11}")
        for sp in spreads:
            fed, pcl = [], []
            closed = None
            for sd in range(args.seeds):
                data, cs, ws = build_clients(values, 0.0, sp, args.M, args.N,
                                             args.alpha, args.se, args.sx, seed=2000 + sd)
                p_hat = fedavg_train(data, n_rounds=args.n_rounds, seed=sd)
                fed.append(leakage_of(p_hat))
                pcl.append(per_client_leakage(data))
                if closed is None:
                    closed = scm.closed_form_leakage(cs, ws)
            fed = np.array(fed)
            rows.append(("b", sp, fed.mean(), fed.std(), closed, float(np.mean(pcl))))
            print(f"{sp:6.2f} {fed.mean():7.4f}+-{fed.std():.4f} {closed:12.4f} {np.mean(pcl):11.4f}")

        # ---- Save intermediate rows as .npz ----
        np.savez(npz_path, rows=np.array(rows, dtype=object))
        print(f"\nSaved intermediate rows to {npz_path}")

    # ---- Save CSV ----
    csv_path = os.path.join(args.outdir, "semisynthetic_curve.csv")
    with open(csv_path, "w") as f:
        f.write("panel,x,fed_mean,fed_std,closed_form,per_client_mean\n")
        for r in rows:
            f.write("%s,%.4f,%.6f,%.6f,%.6f,%.6f\n" % r)
    print(f"Saved {csv_path}")

    # ---- Figure (identical layout to standalone) ----
    A = [r for r in rows if r[0] == "a"]
    B = [r for r in rows if r[0] == "b"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    xa = [r[1] for r in A]
    ax1.errorbar(xa, [r[2] for r in A], yerr=[r[3] for r in A], fmt="o-",
                 capsize=8, label="Ours", color="C0")
    ax1.plot(xa, [r[4] for r in A], "--", color="C0", alpha=0.7,
             label="Fed optimum")
    ax1.plot(xa, [r[5] for r in A], "s-", color="C3",
             label="Local")
    ax1.set_xlabel(r"$\bar{c}$", fontsize=19)
    ax1.set_ylabel(r"$|p_s|/\|p\|$", fontsize=19)
    ax1.set_title(r"(a)", fontsize=20, y=-0.35, fontweight="bold")
    ax1.tick_params(labelsize=16)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=16)

    xb = [r[1] for r in B]
    ax2.errorbar(xb, [r[2] for r in B], yerr=[r[3] for r in B], fmt="o-",
                 capsize=8, color="C0", label=r"Ours")
    ax2.plot(xb, [r[5] for r in B], "s-", color="C3", label="Local")
    ax2.axhline(0.0, color="gray", ls=":", lw=1)
    
    
    
    
    
    
    
    
    
    ax2.set_xlabel(r"$\delta$", fontsize=19)
    ax2.set_ylabel(r"$|p_s|/\|p\|$", fontsize=19)
    ax2.set_title(r"(b)", fontsize=20, y=-0.35, fontweight="bold")
    ax2.tick_params(labelsize=16)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=16)

    fig.tight_layout()
    fig_path_pdf = os.path.join(args.outdir, "fig_invariance_semisynthetic.pdf")
    fig_path_png = os.path.join(args.outdir, "fig_invariance_semisynthetic.png")
    fig.savefig(fig_path_pdf, dpi=150)
    fig.savefig(fig_path_png, dpi=150)
    print(f"Saved {fig_path_pdf}")
    print(f"Saved {fig_path_png}")


if __name__ == "__main__":
    main()
