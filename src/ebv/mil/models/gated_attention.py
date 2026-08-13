"""Gated-attention deep MIL (Ilse et al. 2018) with an MLP head.

Pool patches by masked gated attention, then predict once on the bag embedding:
`p(x) = psi(sum_i alpha_i f(x_i))`. Shapes: `b` bags, `n` patches, `d`=1536
(UNI2-h), `h`=hidden_dim, `c`=n_classes.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ebv.mil.models.registry import register


@register("gated_attention")
class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, attn_dim: int = 256,
                 head_hidden: Sequence[int] = (256,), n_classes: int = 1, dropout: float = 0.25) -> None:
        super().__init__()
        self.input_dim, self.hidden_dim, self.attn_dim = input_dim, hidden_dim, attn_dim
        self.head_hidden, self.n_classes, self.dropout = tuple(head_hidden), n_classes, dropout

        self.ff1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_a = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.projection = nn.Linear(attn_dim, 1)

        layers: list[nn.Module] = []
        width = hidden_dim
        for out in self.head_hidden:
            layers += [nn.Linear(width, out), nn.ReLU(), nn.Dropout(dropout)]
            width = out
        layers.append(nn.Linear(width, n_classes))
        self.classifier = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def config(self) -> dict:
        return {"arch": self.arch, "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
                "attn_dim": self.attn_dim, "head_hidden": list(self.head_hidden),
                "n_classes": self.n_classes, "dropout": self.dropout}

    def project_and_attend(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """-> (attention `b n` summing to 1 over real patches, projected `b n h`)."""
        h = self.ff1(x)
        scores = self.projection(self.attention_a(h) * self.attention_b(h)).squeeze(-1)
        if mask is not None:
            # -inf before softmax so padded patches take no probability mass.
            scores = scores.masked_fill(~mask, float("-inf"))
            empty = ~mask.any(dim=1)  # keep an all-padding row finite rather than NaN
            if empty.any():
                scores = scores.masked_fill(empty.unsqueeze(1), 0.0)
        attention = torch.softmax(scores, dim=1)
        if mask is not None:
            attention = attention * mask
        return attention, h

    def attend(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        attention, h = self.project_and_attend(x, mask)
        return attention, torch.einsum("bn,bnh->bh", attention, h)

    def forward_with_contributions(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """-> (logits `b c`, attention `b n`, per-patch contributions or None).

        Pooling happens before the head, so a patch has no separable contribution;
        the third element is None. AdditiveMIL overrides this.
        """
        attention, bag = self.attend(x, mask)
        return self.classifier(bag), attention, None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        logits, attention, _ = self.forward_with_contributions(x, mask)
        return logits, attention
