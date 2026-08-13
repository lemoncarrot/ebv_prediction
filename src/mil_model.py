#!/usr/bin/env python3
"""MIL heads over UNI2-h patch embeddings: gated-attention, and Additive MIL.

Two architectures share one featurizer/attention stack and differ only in where
the classifier sits relative to the sum over patches:

`GatedAttentionMIL` (Ilse et al. 2018, gated variant)
    Pool first, then predict once: `p(x) = psi_p(sum_i alpha_i f(x_i))`.
    Interpretation is via the attention weights.

`AdditiveMIL` (Javed et al. 2022, arXiv:2206.01794)
    Predict per patch, then sum: `p(x) = sum_i psi_p(alpha_i f(x_i))`.
    Each `psi_p(alpha_i f(x_i))` is that patch's exact, signed, class-wise
    contribution to the logit, so the slide prediction decomposes additively
    over patches instead of being a non-linear function of them.

Both are built from `MIL_implementation.py`'s sketch, completed for real slides:

* variable bag sizes. Slides differ by two orders of magnitude in patch count, so
  a batch is zero-padded to the longest bag and the attention softmax is masked.
  Without the mask, padding rows would take probability mass away from real
  patches and every bag in the batch would be scored on a different effective
  temperature.
* the attention weights are returned alongside the logits. For the gated model
  they are the whole interpretability story -- one scalar per patch, summing to
  one over a slide, which is what `08_attention_heatmaps.py` plots. The additive
  model returns per-patch contributions as well, and those are what its heatmaps
  should use.
* an MLP head rather than a bare linear layer.

Shapes throughout: `b` bags, `n` patches, `d` = 1536 (UNI2-h), `h` = hidden_dim,
`c` = n_classes.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class GatedAttentionMIL(nn.Module):
    """Attention-based deep MIL (Ilse et al. 2018, gated variant) + MLP head.

    Args:
        input_dim: patch embedding width (1536 for UNI2-h).
        hidden_dim: width of the per-patch projection that gets pooled.
        attn_dim: width of the two attention branches.
        head_hidden: hidden widths of the MLP classifier on the bag embedding.
            An empty sequence gives a plain linear classifier.
        n_classes: 1 for binary EBV+/- (a single logit, use BCEWithLogitsLoss).
        dropout: applied after the patch projection and inside the head.
    """

    arch = "gated_attention"

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        head_hidden: Sequence[int] = (256,),
        n_classes: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.head_hidden = tuple(head_hidden)
        self.n_classes = n_classes
        self.dropout = dropout

        self.ff1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
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

    def config(self) -> dict[str, object]:
        """Everything needed to rebuild this model from a checkpoint."""
        return {
            "arch": self.arch,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "attn_dim": self.attn_dim,
            "head_hidden": list(self.head_hidden),
            "n_classes": self.n_classes,
            "dropout": self.dropout,
        }

    def project_and_attend(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project patches and score them by gated attention, without pooling.

        Args:
            x: `b n d` patch embeddings.
            mask: `b n` boolean, True for real patches. None means all real.
        Returns:
            attention: `b n`, non-negative, sums to 1 over real patches.
            h: `b n h` projected patches, i.e. `f(x_i)`.
        """
        h = self.ff1(x)
        scores = self.projection(self.attention_a(h) * self.attention_b(h)).squeeze(-1)

        if mask is not None:
            # -inf before the softmax, not zeroing after it: padded patches must
            # not consume probability mass that belongs to real ones.
            scores = scores.masked_fill(~mask, float("-inf"))
            # An empty bag would make the whole row NaN; keep it finite and let
            # the caller drop it. In practice preprocessing never emits one.
            empty = ~mask.any(dim=1)
            if empty.any():
                scores = scores.masked_fill(empty.unsqueeze(1), 0.0)

        attention = torch.softmax(scores, dim=1)
        if mask is not None:
            attention = attention * mask
        return attention, h

    def attend(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project patches and pool them by gated attention.

        Returns:
            attention: `b n`, non-negative, sums to 1 over real patches.
            bag: `b h` attention-weighted sum of the projected patches.
        """
        attention, h = self.project_and_attend(x, mask)
        bag = torch.einsum("bn,bnh->bh", attention, h)
        return attention, bag

    def forward_with_contributions(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns (logits `b c`, attention `b n`, per-patch contributions or None).

        Pooling happens before the classifier here, so a patch has no separable
        contribution to the logit and the third element is None. `AdditiveMIL`
        overrides this.
        """
        attention, bag = self.attend(x, mask)
        return self.classifier(bag), attention, None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits `b n_classes`, attention `b n`)."""
        logits, attention, _ = self.forward_with_contributions(x, mask)
        return logits, attention


class AdditiveMIL(GatedAttentionMIL):
    """Additive MIL (Javed et al. 2022, arXiv:2206.01794), gated-attention variant.

    Identical to `GatedAttentionMIL` except for the order of the last two
    operations. The attention model computes `psi_p(sum_i alpha_i f(x_i))`; this
    computes `sum_i psi_p(alpha_i f(x_i))`, so the classifier runs per patch and
    the sum happens on logits. The paper calls this "a simple change in function
    composition", and it is: the weights and shapes are unchanged, and any
    pooling-based MIL model can be converted the same way.

    What that buys, and why it is worth a second architecture here: the bag logit
    becomes an exact sum of per-patch terms. Each `psi_p(alpha_i f(x_i))` is that
    patch's marginal contribution -- signed, so a patch can argue *against* the
    positive class, and per class, which a single attention weight cannot express.
    Attention weights only say a patch was used, not what it was used for or in
    which direction, because attention enters the prediction non-linearly.

    Args:
        additive_pool: how per-patch logits combine.

            `mean` (default here) divides by the bag size: `1/N sum_i psi_p(...)`.
            `sum` is the paper's Equation 4 verbatim.

            The default deviates from the paper deliberately. A sum over patches
            is not invariant to bag size: `psi_p` has a bias, so even a patch
            whose `alpha_i f(x_i)` is ~0 still contributes ~`psi_p(0)`, and the
            logit drifts with N. The paper never meets this -- its reported patch
            and bag counts work out to exactly 48 patches per bag on all three of
            its datasets, bag size is a cross-validated constant, and inference
            splits a slide into many same-size bags aggregated by majority vote.
            Our bags are whole slides of ~200 to ~40,000 patches, and
            `07_train_mil.py` trains on a 4096-patch subsample but evaluates on
            the full bag, so the same slide would be scored at two different
            logit scales. Dividing by N removes that drift and costs nothing
            interpretively: the decomposition stays exact, stays signed, stays
            per-class, and the paper's Theorem 1 states *proportionality* to
            Shapley values, which a constant 1/N preserves.

            Use `sum` to reproduce the paper, ideally with fixed-size bags.
    """

    arch = "additive"

    def __init__(self, *args, additive_pool: str = "mean", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if additive_pool not in ("mean", "sum"):
            raise ValueError(f"additive_pool must be 'mean' or 'sum', got {additive_pool!r}")
        self.additive_pool = additive_pool

    def config(self) -> dict[str, object]:
        return {**super().config(), "additive_pool": self.additive_pool}

    def forward_with_contributions(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits `b c`, attention `b n`, contributions `b n c`).

        `logits` is exactly `contributions.sum(dim=1)`; that identity is the
        whole point, and `_selftest` asserts it.
        """
        attention, h = self.project_and_attend(x, mask)
        weighted = attention.unsqueeze(-1) * h  # m_i(x) = alpha_i f(x_i)
        contributions = self.classifier(weighted)  # psi_p(m_i(x)), per class

        if mask is not None:
            # psi_p(0) is the head's bias, not 0, so padded patches would each
            # add a bias term to the sum. Zero them before anything is summed.
            contributions = contributions * mask.unsqueeze(-1)

        if self.additive_pool == "mean":
            if mask is not None:
                counts = mask.sum(dim=1).clamp(min=1).to(contributions.dtype)
            else:
                counts = torch.full((x.shape[0],), float(x.shape[1]),
                                    dtype=contributions.dtype, device=x.device)
            # Scale the contributions themselves, so they still sum to the logit.
            contributions = contributions / counts.view(-1, 1, 1)

        return contributions.sum(dim=1), attention, contributions

    def patch_contributions(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-patch, per-class contributions `b n c`, summing to the bag logits."""
        return self.forward_with_contributions(x, mask)[2]


ARCHITECTURES: dict[str, type[GatedAttentionMIL]] = {
    GatedAttentionMIL.arch: GatedAttentionMIL,
    AdditiveMIL.arch: AdditiveMIL,
}


def build_model(config: dict[str, object]) -> GatedAttentionMIL:
    """Rebuild a model from the dict stored in a checkpoint.

    `arch` defaults to the gated attention model so checkpoints written before
    Additive MIL existed still load.
    """
    arch = str(config.get("arch", GatedAttentionMIL.arch))
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(ARCHITECTURES)}")
    kwargs: dict[str, object] = dict(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        attn_dim=int(config["attn_dim"]),
        head_hidden=tuple(config["head_hidden"]),
        n_classes=int(config["n_classes"]),
        dropout=float(config["dropout"]),
    )
    if arch == AdditiveMIL.arch:
        kwargs["additive_pool"] = str(config.get("additive_pool", "mean"))
    return ARCHITECTURES[arch](**kwargs)


def _selftest() -> None:
    torch.manual_seed(0)
    model = GatedAttentionMIL()
    x = torch.randn(3, 7, 1536)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[1, 4:] = False  # a shorter bag
    logits, attention = model(x, mask)
    assert logits.shape == (3, 1), logits.shape
    assert attention.shape == (3, 7), attention.shape
    assert torch.allclose(attention.sum(1), torch.ones(3), atol=1e-5)
    assert attention[1, 4:].abs().max() == 0

    # Padding must not change a bag's prediction.
    model.eval()
    with torch.no_grad():
        short, _ = model(x[1:2, :4], torch.ones(1, 4, dtype=torch.bool))
        padded, _ = model(x[1:2], mask[1:2])
    assert torch.allclose(short, padded, atol=1e-5), (short, padded)

    # ------------------------------------------------------------- Additive MIL
    torch.manual_seed(0)
    additive = AdditiveMIL(n_classes=2).eval()
    with torch.no_grad():
        add_logits, add_attention, contributions = additive.forward_with_contributions(x, mask)
    assert add_logits.shape == (3, 2), add_logits.shape
    assert contributions.shape == (3, 7, 2), contributions.shape
    assert torch.allclose(add_attention.sum(1), torch.ones(3), atol=1e-5)

    # The property the architecture exists for: the bag logit is exactly the sum
    # of its per-patch contributions.
    assert torch.allclose(contributions.sum(dim=1), add_logits, atol=1e-6)
    # Padded patches contribute exactly nothing, bias included.
    assert contributions[1, 4:].abs().max() == 0

    # Padding invariance again, which for mean pooling also proves the divisor is
    # the real patch count and not the padded width.
    with torch.no_grad():
        short, _, _ = additive.forward_with_contributions(x[1:2, :4],
                                                          torch.ones(1, 4, dtype=torch.bool))
        padded, _, _ = additive.forward_with_contributions(x[1:2], mask[1:2])
    assert torch.allclose(short, padded, atol=1e-5), (short, padded)

    # Bag-size behaviour, the reason `mean` is the default. Biases initialise to
    # zero, which hides the effect, so give the head the non-zero bias a trained
    # model would have; then a sum over N patches drifts with N and a mean does
    # not. Compare bags of 64 and 2048 patches drawn from one distribution.
    for pool, drifts in (("sum", True), ("mean", False)):
        torch.manual_seed(0)
        sized = AdditiveMIL(n_classes=1, additive_pool=pool).eval()
        nn.init.constant_(sized.classifier[-1].bias, 0.5)
        with torch.no_grad():
            small = sized(torch.randn(1, 64, 1536))[0].item()
            large = sized(torch.randn(1, 2048, 1536))[0].item()
        ratio = abs(large) / max(abs(small), 1e-9)
        if drifts:
            assert ratio > 8, f"sum pooling should scale with bag size, got {ratio:.2f}"
        else:
            assert 0.5 < ratio < 2.0, f"mean pooling should be bag-size stable, got {ratio:.2f}"

    # Checkpoint round-trip, including the pre-Additive configs that have no arch.
    assert isinstance(build_model(additive.config()), AdditiveMIL)
    assert build_model(additive.config()).additive_pool == "mean"
    legacy = {k: v for k, v in model.config().items() if k != "arch"}
    assert type(build_model(legacy)) is GatedAttentionMIL
    print("mil_model selftest ok")


if __name__ == "__main__":
    _selftest()
