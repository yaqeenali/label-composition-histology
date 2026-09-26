#!/usr/bin/env python3
"""Convert the ``.npz`` pools into ``.pt`` bundles.

The first-stage pools are numpy archives (originally written by the
preparatory script ``extract_patch_pool.py``, which is not in this repository;
``pool_from_coords.py`` rebuilds them from Trident output and the released
patch coordinates).
``WSIMILDataset`` loads Trident bundles with ``torch.load`` and expects
``{'data': {'features', 'coords'}, 'meta'}``.  This is the one-line bridge.

    python scripts/pool_npz_to_pt.py --src data/tcga_brca/_pool --dst data/tcga_brca/_pool_pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--keep", type=int, default=0,
                   help="Optionally thin further to this many patches, seeded per "
                        "slide. 0 keeps the pool as extracted.")
    p.add_argument("--seed", type=int, default=20260906)
    args = p.parse_args(argv)

    import zlib
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        feats, coords = d["features"], d["coords"]
        if args.keep and feats.shape[0] > args.keep:
            rng = np.random.default_rng(
                (zlib.crc32((str(args.keep) + f.stem).encode()) ^ args.seed) % (2 ** 32))
            idx = np.sort(rng.choice(feats.shape[0], size=args.keep, replace=False))
            feats, coords = feats[idx], coords[idx]
        torch.save({"data": {"features": torch.from_numpy(feats.copy()),
                             "coords": torch.from_numpy(coords.copy())},
                    "meta": {"pool": {"kept": int(feats.shape[0]),
                                      "source_n": int(d["source_n"]),
                                      "encoder": "uni_v1", "name": str(d["name"])}}},
                   dst / (f.stem + ".pt"))
        n += 1
    print(f"converted {n} bundles to {dst}")


if __name__ == "__main__":
    main()
