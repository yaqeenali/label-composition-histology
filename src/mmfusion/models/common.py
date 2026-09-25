"""Shared building blocks for the fusion architectures.

.. warning::

   :class:`PredictionHead` ends in ``nn.Tanh()`` *after* the output projection,
   bounding every prediction to [-1, 1]. For a 0-100 recurrence score the target
   is unreachable; for classification it clamps logits so sigmoid outputs sit in
   roughly [0.27, 0.73]. This is reproduced exactly — all W&B runs and the
   CHIMERA checkpoints were trained with it, so removing it would invalidate
   them. See ``REFACTOR_NUMERICS.md`` entry N-07.

MODULE CONSTRUCTION ORDER IS LOAD-BEARING. Every ``nn.Linear`` draws from the
global torch RNG at construction time, so the order in which submodules are
created determines the initial weights of the whole model. The iteration order
over ``modality_dims`` / ``modality_types`` below is therefore preserved
verbatim from ``model_common.py``.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

__all__ = [
    "AttentionPooling",
    "PredictionHead",
    "ModalityProjectors",
    "apply_modality_dropout",
]


class AttentionPooling(nn.Module):
    """Weighted average pooling with learned per-token attention scores.

    Input  : [B, N, D]  (mask: [B, N] bool, True = valid token)
    Output : [B, D]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn = self.score(x).squeeze(-1)               # [B, N]
        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
        w = torch.softmax(attn, dim=1)                 # [B, N]
        return torch.sum(x * w.unsqueeze(-1), dim=1)   # [B, D]


class PredictionHead(nn.Module):
    """Two-layer MLP head shared by all task types.

    survival / scalar regression -> output_dim=1, returns [B]
    multi-target regression      -> output_dim=n, returns [B, n]
    classification               -> output_dim=n_classes, returns [B, n]
    """

    def __init__(self, d_model: int, output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        hidden = max(d_model // 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
            nn.Tanh(),  # BUG-PRESERVED: bounds the model output to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.head(z)                # [B, output_dim]
        if out.shape[-1] == 1:
            out = out.squeeze(-1)         # [B]
        return out


class ModalityProjectors(nn.Module):
    """One projector per modality, plus attention pooling for bag modalities.

    Each modality gets ``nn.Linear(dim_in, d_model)`` unless overridden through
    ``custom_projectors`` (used to inject a ``ClinicalEmbedder``).
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        modality_types: Dict[str, str],
        d_model: int,
        custom_projectors: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.types = modality_types
        custom = custom_projectors or {}
        # RNG-ORDER: projectors are built in modality_dims order, then pools in
        # modality_types order. Do not reorder.
        self.proj = nn.ModuleDict({
            name: (custom[name] if name in custom else nn.Linear(dim, d_model))
            for name, dim in modality_dims.items()
        })
        self.pool = nn.ModuleDict({
            name: AttentionPooling(d_model)
            for name, t in modality_types.items()
            if t == "bag"
        })

    def forward(
        self,
        modalities: Dict[str, Optional[torch.Tensor]],
        masks: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Project and pool every available modality; skip the missing ones."""
        masks = masks or {}
        out: Dict[str, torch.Tensor] = {}
        for name, x in modalities.items():
            if x is None or name not in self.proj:
                continue
            if self.types.get(name) == "bag":
                h = self.proj[name](x)                # [B, N, D]
                mask = masks.get(name)
                out[name] = self.pool[name](h, mask)  # [B, D]
            else:
                out[name] = self.proj[name](x)        # [B, D]
        return out


def apply_modality_dropout(
    modalities: Dict[str, torch.Tensor],
    p: float,
    training: bool,
) -> Dict[str, torch.Tensor]:
    """Zero out random modalities per sample with probability *p*.

    Rescales surviving modalities by 1/(1-p) to keep expected magnitude
    constant. Draws one ``torch.rand`` per modality per forward pass — the draw
    count and order are part of the RNG contract.
    """
    if not training or p <= 0.0:
        return modalities
    scale = 1.0 / max(1.0 - p, 1e-6)
    out: Dict[str, torch.Tensor] = {}
    for k, v in modalities.items():
        keep = (torch.rand(v.shape[0], 1, device=v.device) >= p).float()
        out[k] = v * keep * scale
    return out
