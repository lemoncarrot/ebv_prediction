"""MIL architectures, discoverable by name. Import the modules so they register."""

from ebv.mil.models.registry import ARCHITECTURES, build_model, register
from ebv.mil.models.gated_attention import GatedAttentionMIL
from ebv.mil.models.additive import AdditiveMIL

__all__ = ["ARCHITECTURES", "build_model", "register", "GatedAttentionMIL", "AdditiveMIL"]
