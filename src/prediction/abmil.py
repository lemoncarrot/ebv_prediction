"""Attention-based deep MIL (Ilse et al. 2018), one bag at a time.

    attention:  p(y) = psi( sum_i a_i * h_i )      h_i = f(x_i),  a = softmax over tiles
    additive:   p(y) = sum_i a_i * psi( h_i )      (Javed et al. 2022; exact tile contributions)
    graph layers (optional): h_i mixed with its 8 grid neighbours before attention
                            (GraphSAGE-style, Patch-GCN-like spatial context)

The module owns the aggregation: attention pooling lives inside `forward`, so a
bag of any length goes in and one logit comes out. It knows nothing about
cohorts, folds or files -- see cohort.py for everything upstream of this.

Shapes, throughout: `n` tiles, `d` input dim (1536 for UNI2-h), `h` hidden.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def logodds(logits: torch.Tensor, positive: int = 1) -> torch.Tensor:
    """Log-odds of class `positive` against all others, over the last dim.

    A single-logit head (..., 1) is already binary and is returned as is, so
    sigmoid(logodds(logits)) is P(positive) for either kind of head.
    """
    if logits.shape[-1] == 1:
        return logits[..., 0]
    rest = torch.cat([logits[..., :positive], logits[..., positive + 1:]], dim=-1)
    return logits[..., positive] - torch.logsumexp(rest, dim=-1)


class GraphLayer(nn.Module):
    """One round of neighbourhood mixing on the tile grid (GraphSAGE-style, mean).

        h_i <- h_i + Dropout(ReLU(W_self h_i + W_nbr mean_{j in N(i)} h_j))

    `neighbours` is (n, 8) tile indices, -1 where a grid neighbour is absent
    (cohort.grid_neighbours). Residual, so stacked layers widen the context
    without washing out each tile's own features.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.nbr_lin = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, neighbours: torch.Tensor) -> torch.Tensor:
        valid = neighbours >= 0                                          # (n, 8)
        gathered = h[neighbours.clamp(min=0)] * valid.unsqueeze(-1)      # (n, 8, d)
        mean = gathered.sum(1) / valid.sum(1, keepdim=True).clamp(min=1)  # isolated tile: 0
        return h + self.drop(torch.relu(self.self_lin(h) + self.nbr_lin(mean)))


class ABMIL(nn.Module):
    """Gated-attention MIL over one bag of tile embeddings.

    forward(x: (n, d)) -> (logits: (n_classes,), attention: (n,))

    The attention weights sum to 1 over tiles and are returned for heatmaps and
    for inspecting which tiles drove a call.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        head_hidden: int | None = 256,
        n_classes: int = 1,
        dropout: float = 0.25,
        aggregation: str = "attention",
        graph_layers: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim, self.hidden_dim, self.attn_dim = input_dim, hidden_dim, attn_dim
        self.head_hidden, self.n_classes, self.dropout = head_hidden, n_classes, dropout
        if aggregation not in ("attention", "additive"):
            raise ValueError(f"aggregation must be attention or additive, got {aggregation!r}")
        self.aggregation = aggregation
        self.graph_layers = int(graph_layers)

        # per-tile projection
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # optional spatial context: graph layers on the tile grid, before attention
        self.graph = nn.ModuleList(GraphLayer(hidden_dim, dropout) for _ in range(self.graph_layers))

        # gated attention: a tanh branch scores, a sigmoid branch gates it
        self.attn_tanh = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attn_gate = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attn_score = nn.Linear(attn_dim, 1)

        # bag-level head
        if head_hidden is None:
            self.head = nn.Linear(hidden_dim, n_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, n_classes),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def attend(self, x: torch.Tensor, neighbours: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """(n, d) -> (attention (n,), projected tiles (n, h)); with graph layers the
        tiles are neighbourhood-mixed first, which needs `neighbours` (n, 8)."""
        h = self.encoder(x)
        if self.graph_layers:
            if neighbours is None:
                raise ValueError("graph layers need the (n, 8) neighbour index for the bag")
            for layer in self.graph:
                h = layer(h, neighbours)
        scores = self.attn_score(self.attn_tanh(h) * self.attn_gate(h))  # (n, 1)
        attention = torch.softmax(scores, dim=0).squeeze(-1)             # (n,)
        return attention, h

    def forward(self, x: torch.Tensor, neighbours: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"expected one bag of shape (n_tiles, {self.input_dim}), "
                             f"got {tuple(x.shape)} -- use DataLoader(batch_size=None)")
        if x.shape[0] == 0:
            raise ValueError("empty bag: filter these out with min_tiles in build_cohort")

        attention, h = self.attend(x, neighbours)
        if self.aggregation == "additive":
            # Additive MIL (Javed et al. 2022), weighted on the scores: classify
            # every tile, then take the attention-weighted sum of tile logits.
            # Bounded (weights sum to 1); contributions a_i * g(h_i) are exact.
            return attention @ self.head(h), attention
        bag = attention @ h              # (h,) -- the aggregation step
        return self.head(bag), attention

    def config(self) -> dict:
        """Enough to rebuild this model from a checkpoint."""
        return {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
                "attn_dim": self.attn_dim, "head_hidden": self.head_hidden,
                "n_classes": self.n_classes, "dropout": self.dropout,
                "aggregation": self.aggregation, "graph_layers": self.graph_layers}

    @classmethod
    def from_config(cls, config: dict) -> "ABMIL":
        return cls(**config)
