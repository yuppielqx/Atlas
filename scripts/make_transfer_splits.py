#!/usr/bin/env python
"""Generate source + complement (rest) client-ID files for transfer experiments.

For each (dataset, fraction) pair, produces:
  - client_ids/<dataset>/frac{XX}_n{N}.txt      (source subset)
  - client_ids/<dataset>/rest_frac{XX}_n{N}.txt  (complement: all nodes \ source)

Uses seed=42 for reproducibility.  For METR-LA the source files already exist;
this script regenerates them identically and adds the missing rest files.
"""

import argparse
import json
import os
import sys

import numpy as np

DATASET_CONFIG = {
    "metr-la": {"dir": "datasets/METR-LA", "total": 207},
    "pems-bay": {"dir": "datasets/PEMS-BAY", "total": 325},
}

FRACTIONS = [0.05, 0.25, 0.50, 0.75, 0.90]
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="metr-la,pems-bay",
                   help="comma-separated dataset keys (default: metr-la,pems-bay)")
    p.add_argument("--fractions", default="0.05,0.25,0.50,0.75,0.90",
                   help="comma-separated source fractions")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be written, do not write")
    return p.parse_args()


def write_ids(path, ids, dry_run):
    """Write newline-separated integer node IDs to a text file."""
    if dry_run:
        print(f"  [dry-run] would write {len(ids)} ids to {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(str(int(x)) for x in ids))
        f.write("\n")


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]

    for ds in datasets:
        if ds not in DATASET_CONFIG:
            print(f"ERROR: unknown dataset '{ds}'. Known: {list(DATASET_CONFIG.keys())}",
                  file=sys.stderr)
            continue
        cfg = DATASET_CONFIG[ds]

        # Read actual node count from desc.json if available
        desc_path = os.path.join(cfg["dir"], "desc.json")
        if os.path.exists(desc_path):
            with open(desc_path) as f:
                desc = json.load(f)
            num_nodes = int(desc.get("num_nodes") or desc["shape"][1])
        else:
            num_nodes = cfg["total"]

        out_dir = f"client_ids/{ds}"
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(num_nodes)

        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  |  Total nodes: {num_nodes}  |  Seed: {args.seed}")
        print(f"{'='*60}")

        for frac in fractions:
            n_src = int(round(num_nodes * frac))
            n_src = max(1, min(num_nodes, n_src))
            n_rest = num_nodes - n_src
            tag = f"frac{int(round(frac * 100)):02d}_n{n_src}"

            src_ids = np.sort(perm[:n_src])
            rest_ids = np.sort(perm[n_src:])

            src_path = os.path.join(out_dir, f"{tag}.txt")
            rest_path = os.path.join(out_dir, f"rest_{tag}.txt")

            write_ids(src_path, src_ids, args.dry_run)
            write_ids(rest_path, rest_ids, args.dry_run)

            # Verify no overlap and complete coverage
            src_set = set(src_ids)
            rest_set = set(rest_ids)
            assert len(src_set) == n_src, f"Source duplicate detected for {ds}/{tag}"
            assert len(rest_set) == n_rest, f"Rest duplicate detected for {ds}/{tag}"
            assert src_set.isdisjoint(rest_set), \
                f"Source/rest overlap for {ds}/{tag}: {src_set & rest_set}"
            assert len(src_set | rest_set) == num_nodes, \
                f"Missing nodes for {ds}/{tag}: {num_nodes - len(src_set | rest_set)}"

            print(f"  {tag}: {n_src:4d} source + {n_rest:4d} rest = {num_nodes} ✓")

    print(f"\nDone. {'[dry-run — no files written]' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
