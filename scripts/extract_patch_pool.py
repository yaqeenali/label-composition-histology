#!/usr/bin/env python3
"""Extract a fixed, seeded pool of patches from each Trident feature bundle.

Why this exists
---------------
The locked re-run needs one configuration applied identically to all four
assays.  The original notebooks disagreed here too: GHI subsampled 500 patches
per access while the other three used every patch of every slide.  Fixing a
common patch budget is part of pre-specifying the configuration.

Fixing the pool *once*, with a recorded seed, also removes the per-access
``torch.randperm`` draw documented as N-10 -- evaluation becomes deterministic
and a fold's test predictions no longer depend on how many times the loader was
called before them.

Runs without PyTorch: ``torch.save`` writes an uncompressed zip, so the raw
float32 / int32 storages are read directly with numpy.  Output is ``.npz``;
``pool_npz_to_pt.py`` converts to the ``.pt`` layout the dataset expects.
"""
from __future__ import annotations
import argparse, io, json, pickle, sys, zipfile, zlib
from pathlib import Path
import numpy as np

DTYPES = {"FloatStorage": np.float32, "IntStorage": np.int32,
          "LongStorage": np.int64, "DoubleStorage": np.float64,
          "HalfStorage": np.float16}


def _rebuild(storage, offset, size, stride, *rest):
    return {"kind": "tensor", "storage": storage, "offset": offset,
            "size": size, "stride": stride}


class _Any:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): self.state = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == "_rebuild_tensor_v2":
            return _rebuild
        return type(name, (_Any,), {})

    def persistent_load(self, pid):
        return {"dtype": getattr(pid[1], "__name__", str(pid[1])),
                "key": pid[2], "numel": pid[4]}


def read_bundle(path: Path):
    """Return (features[N,1024] float32, coords[N,2] int32) without torch."""
    with zipfile.ZipFile(path) as z:
        root = z.namelist()[0].split("/")[0]
        obj = _Unpickler(io.BytesIO(z.read(f"{root}/data.pkl"))).load()
        out = {}
        for key in ("features", "coords"):
            t = obj["data"][key]
            st = t["storage"]
            raw = z.read(f"{root}/data/{st['key']}")
            arr = np.frombuffer(raw, dtype=DTYPES[st["dtype"]], count=st["numel"])
            out[key] = arr.reshape(t["size"])
        return out["features"], out["coords"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="directory of *.pt bundles")
    p.add_argument("--dst", required=True, help="output directory for *.npz")
    p.add_argument("--pool", type=int, default=1024, help="patches kept per slide")
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    args = p.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.pt"))
    sel = files[args.start:] if not args.limit else files[args.start:args.start + args.limit]

    manifest = []
    for f in sel:
        out = dst / (f.stem + ".npz")
        if out.exists():
            continue
        feats, coords = read_bundle(f)
        n = feats.shape[0]
        # Seed per slide from the slide NAME so the pool is reproducible
        # regardless of the order or batching this script is run in.
        # crc32, not hash(): Python randomises string hashing per process, which
        # would have made this silently irreproducible across runs.
        rng = np.random.default_rng(
            (zlib.crc32(f.stem.encode()) ^ args.seed) % (2**32))
        if n > args.pool:
            idx = np.sort(rng.choice(n, size=args.pool, replace=False))
        else:
            idx = np.arange(n)
        np.savez(out, features=feats[idx].astype(np.float32),
                 coords=coords[idx].astype(np.int32),
                 source_n=np.int64(n), kept=np.int64(len(idx)), name=f.stem)
        manifest.append({"slide": f.stem, "source_patches": int(n), "kept": int(len(idx))})
        print(f"{f.stem[:44]}  {n:6d} -> {len(idx)}", flush=True)

    mf = dst / "pool_manifest.jsonl"
    with mf.open("a") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")
    print(f"\ndone: {len(manifest)} written to {dst}")


if __name__ == "__main__":
    main()
