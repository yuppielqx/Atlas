#!/usr/bin/env python
"""Write nested node-id subsets for source-client transfer experiments."""

import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="datasets/METR-LA",
                   help="dataset folder containing desc.json")
    p.add_argument("--out-dir", default="client_ids/metr-la",
                   help="directory for generated *.txt files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fractions", default="0.05,0.25,0.50,0.75,0.90",
                   help="comma-separated source fractions")
    return p.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.dataset_dir, "desc.json")) as f:
        desc = json.load(f)
    num_nodes = int(desc.get("num_nodes") or desc["shape"][1])
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(num_nodes)
    os.makedirs(args.out_dir, exist_ok=True)

    for frac in fractions:
        n = int(round(num_nodes * frac))
        n = max(1, min(num_nodes, n))
        ids = np.sort(perm[:n])
        tag = f"frac{int(round(frac * 100)):02d}_n{n}"
        path = os.path.join(args.out_dir, f"{tag}.txt")
        with open(path, "w") as f:
            f.write("\n".join(str(int(x)) for x in ids))
            f.write("\n")
        print(f"{tag}: {path}")


if __name__ == "__main__":
    main()
