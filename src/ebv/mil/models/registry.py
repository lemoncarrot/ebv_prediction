"""Architecture registry. Wire in a new MIL head by decorating it with @register."""

from __future__ import annotations

ARCHITECTURES: dict[str, type] = {}


def register(name: str):
    """Register an nn.Module under `name` and set its `.arch` attribute."""
    def decorator(cls):
        ARCHITECTURES[name] = cls
        cls.arch = name
        return cls
    return decorator


def build_model(config: dict):
    """Rebuild a model from the dict a checkpoint stored. `arch` defaults to
    gated_attention so pre-registry checkpoints still load."""
    arch = str(config.get("arch", "gated_attention"))
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(ARCHITECTURES)}")
    cls = ARCHITECTURES[arch]
    kwargs = dict(
        input_dim=int(config["input_dim"]), hidden_dim=int(config["hidden_dim"]),
        attn_dim=int(config["attn_dim"]), head_hidden=tuple(config["head_hidden"]),
        n_classes=int(config["n_classes"]), dropout=float(config["dropout"]),
    )
    if arch == "additive":
        kwargs["additive_pool"] = str(config.get("additive_pool", "mean"))
    return cls(**kwargs)
