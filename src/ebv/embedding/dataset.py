"""Read WSI patches at their stored coordinates and apply the UNI2 transform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import Dataset


class WSIPatchDataset(Dataset):
    """One item = one patch, read at `target_downsample` and resized to `patch_size`.

    The OpenSlide handle is opened lazily so the dataset pickles cleanly into
    DataLoader worker processes.
    """

    def __init__(self, slide_path: Path, coords: np.ndarray, patch_size: int,
                 target_downsample: float, transform) -> None:
        self.slide_path = str(slide_path)
        self.coords = coords
        self.patch_size = patch_size
        self.target_downsample = target_downsample
        self.transform = transform
        self._slide: openslide.OpenSlide | None = None

    def __len__(self) -> int:
        return len(self.coords)

    def _slide_handle(self) -> openslide.OpenSlide:
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.slide_path)
        return self._slide

    def __getitem__(self, index: int) -> torch.Tensor:
        slide = self._slide_handle()
        x, y = (int(v) for v in self.coords[index])
        level = slide.get_best_level_for_downsample(self.target_downsample)
        level_ds = float(slide.level_downsamples[level])
        read = max(1, round(self.patch_size * self.target_downsample / level_ds))
        patch = slide.read_region((x, y), level, (read, read)).convert("RGB")
        if patch.size != (self.patch_size, self.patch_size):
            patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.BICUBIC)
        return self.transform(patch)

    def __del__(self) -> None:
        if self._slide is not None:
            self._slide.close()


def autocast_context(device: str, amp: str):
    """amp is 'none' | 'float16' | 'bfloat16'."""
    dtype = torch.float16 if amp == "float16" else torch.bfloat16
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=amp != "none")
