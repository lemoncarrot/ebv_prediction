"""Additive MIL (Javed et al. 2022, arXiv:2206.01794), gated-attention variant.

Same weights as GatedAttentionMIL, but the classifier runs per patch and the sum
happens on logits: `p(x) = sum_i psi(alpha_i f(x_i))`. The bag logit is then an
exact sum of per-patch terms, so each patch has a signed, per-class contribution
— a patch can argue against the positive class, which one attention weight cannot.
"""

from __future__ import annotations

import torch

from ebv.mil.models.gated_attention import GatedAttentionMIL
from ebv.mil.models.registry import register


@register("additive")
class AdditiveMIL(GatedAttentionMIL):
    def __init__(self, *args, additive_pool: str = "mean", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if additive_pool not in ("mean", "sum"):
            raise ValueError(f"additive_pool must be 'mean' or 'sum', got {additive_pool!r}")
        # mean divides per-patch logits by bag size. psi has a bias, so a raw sum
        # drifts with N — and our bags run ~200 to ~40k patches, trained on a 4096
        # subsample but evaluated whole. mean removes that drift; the decomposition
        # stays exact (Theorem 1 gives proportionality to Shapley, which 1/N keeps).
        # Use sum to reproduce the paper, ideally with fixed-size bags.
        self.additive_pool = additive_pool

    def config(self) -> dict:
        return {**super().config(), "additive_pool": self.additive_pool}

    def forward_with_contributions(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """-> (logits `b c`, attention `b n`, contributions `b n c`).
        logits == contributions.sum(dim=1) exactly."""
        attention, h = self.project_and_attend(x, mask)
        contributions = self.classifier(attention.unsqueeze(-1) * h)  # psi(alpha_i f(x_i)), per class
        if mask is not None:
            contributions = contributions * mask.unsqueeze(-1)  # psi(0) is the bias, not 0
        if self.additive_pool == "mean":
            counts = (mask.sum(dim=1).clamp(min=1).to(contributions.dtype) if mask is not None
                      else torch.full((x.shape[0],), float(x.shape[1]), dtype=contributions.dtype,
                                      device=x.device))
            contributions = contributions / counts.view(-1, 1, 1)  # keeps sum == logit
        return contributions.sum(dim=1), attention, contributions

    def patch_contributions(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_with_contributions(x, mask)[2]
