#!/usr/bin/env python3
"""Check that this machine's UNI2-h setup reproduces the existing embeddings.

    python check_encoder.py --slide ../../data/original_chile/140-A4_150818.svs \
                            --embedding ../../data/chile_embeddings/140-A4_150818.h5

Re-reads N tiles at the coordinates stored in an existing embedding file (same
SlideReader, same tile size and mpp), encodes them with the local UNIEncoder,
and compares with the stored features. Embeddings made here are only usable with
models trained on the old ones if they agree: cosine similarity ~1 (small
differences come from fp16 and library versions). Exits non-zero if the median
cosine is below --min-cosine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SlideReaderClass import SlideReader  # noqa: E402
from encoders.UNI import UNIEncoder  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slide", type=Path, required=True)
    p.add_argument("--embedding", type=Path, required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--min-cosine", type=float, default=0.99)
    args = p.parse_args()

    with h5py.File(args.embedding, "r") as h:
        coords, stored = h["coords"][:], h["features"]
        pick = np.linspace(0, len(coords) - 1, min(args.n, len(coords))).astype(int)
        ref = stored[np.sort(pick)].astype(np.float64)
        old = {k: h.attrs[k] for k in ("encoder", "mpp_target", "patch_size", "amp") if k in h.attrs}
    coords = coords[np.sort(pick)]

    encoder = UNIEncoder()
    with SlideReader(str(args.slide), mpp_target=encoder.spec.mpp_target,
                     tile_size=encoder.spec.tile_size) as reader:
        tiles = np.stack([reader.read_tile(int(x), int(y)) for x, y in coords])
    new = encoder.encode(tiles).astype(np.float64)

    cos = (ref * new).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(new, axis=1))
    rel = np.linalg.norm(ref - new, axis=1) / np.linalg.norm(ref, axis=1)
    print(f"stored attrs: {old}")
    print(f"local encoder: {encoder.spec.name}, amp={encoder.amp}, device={encoder.device}")
    print(f"{len(cos)} tiles: cosine median {np.median(cos):.5f} min {cos.min():.5f}; "
          f"relative L2 median {np.median(rel):.4f}")
    ok = np.median(cos) >= args.min_cosine
    print("MATCH" if ok else "MISMATCH: embeddings from this setup are not comparable")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
