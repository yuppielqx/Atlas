"""
Standalone validation of Proposition 1'(iii):
    spurious leakage of the *shared* representation is proportional to the
    federation-averaged spurious correlation cbar = sum_k w_k c_k, and is
    governed by the MEAN cbar, not by the within-federation spread.

This uses the linear surrogate of CausalFedST: a single shared unit direction
p (the K=1 prototype / projection axis) trained by FedAvg, with a personalized
linear head (a_k, b_k) per client. It reproduces the closed-form federated
optimum derived in theory/prop1_invariance.tex and contrasts it with
per-client (non-federated) ERM.

Data-generating SCM (Assumptions 1-2 of the proposition), per client k:
    u ~ N(0, su^2);  eps ~ N(0, se^2);  y = alpha*u + eps        (invariant)
    xi ~ N(0, sx^2); s = c_k*y + xi                              (spurious)
    z = [u, s]                                                   (observed)

Run:
    python theory/exp_invariance_standalone.py
Outputs:
    theory/out/standalone_curve.csv
    theory/out/fig_invariance_standalone.png   (panel a: leakage vs cbar;
                                                 panel b: control, leakage vs spread)
"""

import os
import argparse
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/causal_fedts_mpl_{os.getuid()}")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- #
#  SCM + closed-form reference                                                   #
# ----------------------------------------------------------------------------- #
class SCM:
    """Population/sample moments for the linear-Gaussian SCM."""

    def __init__(self, alpha=1.0, su=1.0, se=0.5, sx=1.0):
        self.alpha, self.su2, self.se2, self.sx2 = alpha, su**2, se**2, sx**2
        self.VY = alpha**2 * su**2 + se**2

    def sample(self, c_k, N, rng):
        """Draw N samples for a client with spurious strength c_k. Returns z(N,2), y(N,)."""
        u = rng.normal(0.0, np.sqrt(self.su2), N)
        eps = rng.normal(0.0, np.sqrt(self.se2), N)
        y = self.alpha * u + eps
        xi = rng.normal(0.0, np.sqrt(self.sx2), N)
        s = c_k * y + xi
        z = np.stack([u, s], axis=1).astype(np.float32)
        return z, y.astype(np.float32)

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


# ----------------------------------------------------------------------------- #
#  Federated training of the shared direction (FedAvg + personalized heads)      #
# ----------------------------------------------------------------------------- #
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
        p_global = p_global / (p_global.norm() + 1e-12)  # spherical FedAvg
    return p_global.numpy(), float(np.mean(final_losses))


def fedavg_train(client_data, n_rounds=80, local_steps=2, lr=0.05, seed=0, n_restarts=4):
    """FedAvg over a shared unit direction p with personalized linear heads.

    Model per client k:  y_hat = a_k * (z @ p) + b_k.
    p is shared (broadcast -> local SGD -> averaged -> renormalized each round);
    (a_k, b_k) are personalized and persist across rounds.

    Runs ``n_restarts`` random inits and returns the one with the lowest federated
    training loss. This selects the global optimizer (matching the closed-form
    argmax) and avoids spuriously falling into a symmetric local optimum at the
    pathological cbar=0 symmetric point.
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
        # OLS of y on [u, s] (+intercept); direction = normalized coefficient vector.
        A = np.concatenate([z, np.ones((len(z), 1), np.float32)], axis=1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        leaks.append(leakage_of(beta[:2]))
    return float(np.mean(leaks))


# ----------------------------------------------------------------------------- #
#  Experiment driver                                                             #
# ----------------------------------------------------------------------------- #
def build_clients(scm, cbar, spread, M, N, seed):
    """c_k = cbar + spread * r_k with r_k a fixed zero-mean +/-1 pattern."""
    rng = np.random.default_rng(seed)
    rk = np.where(np.arange(M) < M // 2, 1.0, -1.0)  # mean exactly 0
    cs = cbar + spread * rk
    data = [scm.sample(c, N, rng) for c in cs]
    ws = np.full(M, 1.0 / M)
    return data, cs, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--su", type=float, default=1.0)
    ap.add_argument("--se", type=float, default=0.5)
    ap.add_argument("--sx", type=float, default=2.0,
                    help="spurious noise std; larger keeps the invariant direction "
                         "globally dominant so FedAvg reliably finds it")
    ap.add_argument("--spread", type=float, default=0.5, help="fixed spread for panel (a)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_rounds", type=int, default=60)
    ap.add_argument("--outdir", type=str, default="theory/out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    scm =   (args.alpha, args.su, args.se, args.sx)
    print(f"SCM VY={scm.VY:.3f}; sx={args.sx} keeps the invariant direction globally "
          f"dominant so FedAvg reliably finds it")

    cbars = np.round(np.arange(0.0, 1.0001, 0.1), 3)
    rows = []  # (panel, x, fed_mean, fed_std, closed, perclient_mean)

    # ---- Panel (a): leakage vs cbar, spread fixed ----
    print("\n[Panel a] leakage vs cbar (spread=%.2f)" % args.spread)
    print(f"{'cbar':>6} {'fed':>14} {'closed-form':>12} {'per-client':>11}")
    for cbar in cbars:
        fed, pcl = [], []
        closed = None
        for sd in range(args.seeds):
            data, cs, ws = build_clients(scm, cbar, args.spread, args.M, args.N, seed=1000 + sd)
            p_hat = fedavg_train(data, n_rounds=args.n_rounds, seed=sd)
            fed.append(leakage_of(p_hat))
            pcl.append(per_client_leakage(data))
            if closed is None:
                closed = scm.closed_form_leakage(cs, ws)
        fed = np.array(fed)
        rows.append(("a", cbar, fed.mean(), fed.std(), closed, float(np.mean(pcl))))
        print(f"{cbar:6.2f} {fed.mean():7.4f}+-{fed.std():.4f} {closed:12.4f} {np.mean(pcl):11.4f}")

    # ---- Panel (b): control. leakage vs spread, cbar fixed at 0 ----
    spreads = [0.3, 0.6, 0.9, 1.2, 1.5]
    print("\n[Panel b] control: leakage vs spread (cbar=0) -> should stay ~0")
    print(f"{'spread':>6} {'fed':>14} {'closed-form':>12} {'per-client':>11}")
    for sp in spreads:
        fed, pcl = [], []
        closed = None
        for sd in range(args.seeds):
            data, cs, ws = build_clients(scm, 0.0, sp, args.M, args.N, seed=2000 + sd)
            p_hat = fedavg_train(data, n_rounds=args.n_rounds, seed=sd)
            fed.append(leakage_of(p_hat))
            pcl.append(per_client_leakage(data))
            if closed is None:
                closed = scm.closed_form_leakage(cs, ws)
        fed = np.array(fed)
        rows.append(("b", sp, fed.mean(), fed.std(), closed, float(np.mean(pcl))))
        print(f"{sp:6.2f} {fed.mean():7.4f}+-{fed.std():.4f} {closed:12.4f} {np.mean(pcl):11.4f}")

    # ---- Save CSV ----
    csv_path = os.path.join(args.outdir, "standalone_curve.csv")
    with open(csv_path, "w") as f:
        f.write("panel,x,fed_mean,fed_std,closed_form,per_client_mean\n")
        for r in rows:
            f.write("%s,%.4f,%.6f,%.6f,%.6f,%.6f\n" % r)
    print(f"\nSaved {csv_path}")

    # ---- Figure ----
    A = [r for r in rows if r[0] == "a"]
    B = [r for r in rows if r[0] == "b"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    xa = [r[1] for r in A]
    ax1.errorbar(xa, [r[2] for r in A], yerr=[r[3] for r in A], fmt="o-",
                 capsize=3, label="Federated (FedAvg shared $p$)", color="C0")
    ax1.plot(xa, [r[4] for r in A], "--", color="C0", alpha=0.7,
             label="Closed-form federated optimum")
    ax1.plot(xa, [r[5] for r in A], "s-", color="C3",
             label="Per-client (non-federated)")
    ax1.set_xlabel(r"federation-averaged spurious correlation  $\bar{c}$")
    ax1.set_ylabel(r"spurious leakage  $|p_s|/\|p\|$")
    ax1.set_title(r"(a)  leakage $\propto \bar{c}$, vanishes at $\bar{c}=0$")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    xb = [r[1] for r in B]
    ax2.errorbar(xb, [r[2] for r in B], yerr=[r[3] for r in B], fmt="o-",
                 capsize=3, color="C0", label=r"Federated ($\bar{c}=0$ held)")
    ax2.plot(xb, [r[5] for r in B], "s-", color="C3", label="Per-client (non-federated)")
    ax2.axhline(0.0, color="gray", ls=":", lw=1)
    ax2.set_xlabel(r"within-federation spread of $c_k$  (at $\bar{c}=0$)")
    ax2.set_ylabel(r"spurious leakage  $|p_s|/\|p\|$")
    ax2.set_title(r"(b)  control: federation $\to 0$ regardless of spread")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(args.outdir, "fig_invariance_standalone.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
