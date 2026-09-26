#!/usr/bin/env python3
"""Rebuild the first-stage 1,024-patch pools from Trident output and the released coordinates.

The first-stage pools (1,024 patches per slide) were drawn during the
preparatory work with a script that is not part of this repository, so the
draw itself cannot be repeated from it. Instead,
``data/tcga_brca/patch_coords_1024.csv.gz`` lists, for each of the 90 slides,
the level-0 coordinates of the 1,024 patches in their original order
(``pool_index``) and whether each patch is in the 512-patch subset used for
training (``in_512``).

This script reads one Trident feature file per slide (HDF5 with datasets
``features``, N x 1024, and ``coords``, N x 2), picks the listed patches by
their coordinates, in the listed order, and writes the ``.npz`` archives that
``pool_npz_to_pt.py`` converts into training bundles:

    python scripts/pool_from_coords.py --h5_dir <trident>/features_uni_v1 \\
        --coords data/tcga_brca/patch_coords_1024.csv.gz --out data/tcga_brca/_pool
    python scripts/pool_npz_to_pt.py --src data/tcga_brca/_pool \\
        --dst data/tcga_brca/_pool_pt --keep 512

``pool_npz_to_pt.py --keep 512`` then selects exactly the patches flagged
``in_512``. The patch selection is exact; the embeddings are only as close to
the original ones as the re-run of Trident and UNI allows (hardware and
software versions can change the values slightly). Requires ``h5py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5_dir", required=True, help="folder with one Trident <slide>.h5 per slide")
    p.add_argument("--coords", default="data/tcga_brca/patch_coords_1024.csv.gz")
    p.add_argument("--out", required=True, help="folder for the .npz pools")
    a = p.parse_args(argv)
    import h5py

    tab = pd.read_csv(a.coords)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    missing, done = [], 0
    for slide, g in tab.groupby("slide", sort=True):
        g = g.sort_values("pool_index")
        f = Path(a.h5_dir) / f"{slide}.h5"
        if not f.exists():
            missing.append(slide); continue
        with h5py.File(f, "r") as h:
            feats = np.asarray(h["features"])
            coords = np.asarray(h["coords"])
        if feats.ndim == 3:            # accept a leading axis of size 1
            feats = feats[0]
        if coords.ndim == 3:
            coords = coords[0]
        where = {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}
        try:
            idx = np.array([where[(int(x), int(y))] for x, y in zip(g.x, g.y)])
        except KeyError as e:
            raise SystemExit(f"{slide}: coordinate {e} not in {f.name}; the Trident run used "
                             "different segmentation or patching settings")
        np.savez(out / f"{slide}.npz", features=feats[idx].astype(np.float32),
                 coords=coords[idx].astype(np.int32), source_n=np.int64(len(coords)),
                 kept=np.int64(len(idx)), name=np.array(slide))
        done += 1
    print(f"wrote {done} pools to {out}")
    if missing:
        print(f"{len(missing)} slides without an .h5 file, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
