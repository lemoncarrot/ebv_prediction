"""UNI2-h tile encoder.

The contract with the rest of the pipeline is deliberately narrow:

    encode(tiles: uint8 (B, tile_size, tile_size, 3) RGB) -> float32 (B, embed_dim)

which is exactly what `np.stack([reader.read_tile(x, y) for x, y in batch])`
hands over -- uint8, HWC, RGB, alpha already flattened onto white.

Resize/crop/ToTensor/ImageNet-normalize all live inside the transform that
`uni.get_encoder` returns, so none of it is reimplemented here. 
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncoderSpec:
    """Everything the caller needs to sample tiles the encoder expects."""

    name: str
    enc_name: str
    embed_dim: int
    mpp_target: float
    tile_size: int
    model_input_size: int


# tile_size is the *field of view*, not the model input: 256 px at 0.5 MPP is
# 128 um, which the official transform then resizes to 224. Tiling at 224 here
# would silently shrink the field of view to 112 um -- the parameter the TCGA
# reference check is most sensitive to.
SPEC = EncoderSpec(
    name="MahmoodLab/UNI2-h",
    enc_name="uni2-h",
    embed_dim=1536,
    mpp_target=0.5,
    tile_size=256,
    model_input_size=224,
)


class UNIEncoder:
    """UNI2-h, wrapped to take uint8 tile batches and return float32 embeddings."""

    spec = SPEC

    def __init__(self, device: str | None = None, amp: str = "float16") -> None:
        import torch
        from uni import get_encoder

        if amp not in ("none", "float16", "bfloat16"):
            raise ValueError(f"amp must be none/float16/bfloat16, got {amp!r}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not device.startswith("cuda") and amp == "float16":
            amp = "none"  # fp16 autocast on CPU is not useful

        self._torch = torch
        self.device = device
        self.amp = amp

        self.model, self.transform = get_encoder(enc_name=self.spec.enc_name, device=device)
        self.model.eval()

    def _autocast(self):
        if self.amp == "none":
            return contextlib.nullcontext()
        dtype = self._torch.float16 if self.amp == "float16" else self._torch.bfloat16
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        return self._torch.autocast(device_type=device_type, dtype=dtype)

    def _check(self, tiles: np.ndarray) -> None:
        s = self.spec.tile_size
        if tiles.dtype != np.uint8:
            raise TypeError(f"expected uint8 tiles, got {tiles.dtype}")
        if tiles.ndim != 4 or tiles.shape[1:] != (s, s, 3):
            raise ValueError(
                f"expected (B, {s}, {s}, 3) RGB tiles, got {tuple(tiles.shape)}. "
                f"Build the SlideReader with tile_size={s}, mpp_target={self.spec.mpp_target}."
            )

    def encode(self, tiles: np.ndarray) -> np.ndarray:
        """uint8 (B, tile_size, tile_size, 3) RGB -> float32 (B, embed_dim)."""
        from PIL import Image

        self._check(tiles)
        torch = self._torch

        batch = torch.stack([self.transform(Image.fromarray(t)) for t in tiles])
        batch = batch.to(self.device, non_blocking=True)

        with torch.inference_mode(), self._autocast():
            out = self.model(batch)

        return out.float().cpu().numpy().astype(np.float32, copy=False)
