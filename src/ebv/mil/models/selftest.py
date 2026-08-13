#!/usr/bin/env python3
"""Invariant checks for the MIL architectures. Run after wiring in a new one:

    python -m ebv.mil.models.selftest
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn

from ebv.mil.models import AdditiveMIL, GatedAttentionMIL, build_model


def main() -> None:
    torch.manual_seed(0)
    model = GatedAttentionMIL()
    x = torch.randn(3, 7, 1536)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[1, 4:] = False  # a shorter bag

    logits, attention = model(x, mask)
    assert logits.shape == (3, 1) and attention.shape == (3, 7)
    assert torch.allclose(attention.sum(1), torch.ones(3), atol=1e-5)
    assert attention[1, 4:].abs().max() == 0  # padding gets no attention

    # Padding must not change a bag's prediction.
    model.eval()
    with torch.no_grad():
        short, _ = model(x[1:2, :4], torch.ones(1, 4, dtype=torch.bool))
        padded, _ = model(x[1:2], mask[1:2])
    assert torch.allclose(short, padded, atol=1e-5)

    # Additive MIL: the bag logit is exactly the sum of per-patch contributions.
    torch.manual_seed(0)
    additive = AdditiveMIL(n_classes=2).eval()
    with torch.no_grad():
        add_logits, add_attention, contributions = additive.forward_with_contributions(x, mask)
    assert contributions.shape == (3, 7, 2)
    assert torch.allclose(add_attention.sum(1), torch.ones(3), atol=1e-5)
    assert torch.allclose(contributions.sum(dim=1), add_logits, atol=1e-6)
    assert contributions[1, 4:].abs().max() == 0  # padded patches contribute nothing

    # mean pooling is bag-size stable where sum drifts (give the head a real bias).
    for pool, drifts in (("sum", True), ("mean", False)):
        torch.manual_seed(0)
        sized = AdditiveMIL(n_classes=1, additive_pool=pool).eval()
        nn.init.constant_(sized.classifier[-1].bias, 0.5)
        with torch.no_grad():
            small = sized(torch.randn(1, 64, 1536))[0].item()
            large = sized(torch.randn(1, 2048, 1536))[0].item()
        ratio = abs(large) / max(abs(small), 1e-9)
        assert (ratio > 8) if drifts else (0.5 < ratio < 2.0), f"{pool}: ratio {ratio:.2f}"

    # Checkpoint round-trip through the registry, including legacy (no arch) configs.
    assert isinstance(build_model(additive.config()), AdditiveMIL)
    assert build_model(additive.config()).additive_pool == "mean"
    legacy = {k: v for k, v in model.config().items() if k != "arch"}
    assert type(build_model(legacy)) is GatedAttentionMIL
    print("mil models selftest ok")


if __name__ == "__main__":
    main()
