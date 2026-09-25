"""The three multimodal fusion architectures.

Previously one file each (``model_abmil.py``, ``model_modality_fusion.py``,
``model_cross_attention_fusion.py``). They share a constructor signature, a
projector stage and a prediction head, so they now sit together; the classes
themselves are unchanged.

All three follow the same forward shape::

    projectors -> + per-modality token embedding -> modality dropout
               -> fuse -> PredictionHead

RNG-ORDER: submodule construction order is preserved exactly in every class.
Any reordering changes initial weights and therefore every downstream number.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from mmfusion.models.common import (
    ModalityProjectors,
    PredictionHead,
    apply_modality_dropout,
)

__all__ = [
    "ABMILFusionModel",
    "ModalityFusionModel",
    "CrossAttentionFusionModel",
    "CrossAttentionBlock",
    "SelfAttentionBlock",
]


class _FusionBase(nn.Module):
    """Shared projector + modality-token + modality-dropout stage.

    Factored out of the three models' identical forward preamble. It performs
    exactly the operations that were duplicated in each: project, add the
    per-modality embedding for modalities present in ``self.modalities`` order,
    apply modality dropout, and raise if nothing survived.
    """

    modalities: List[str]
    modality_dropout: float

    def _build_common(
        self,
        modalities: List[str],
        modality_dims: Dict[str, int],
        modality_types: Dict[str, str],
        d_model: int,
        modality_dropout: float,
        custom_projectors: Optional[Dict[str, nn.Module]],
    ) -> None:
        self.modalities = modalities
        self.modality_dropout = modality_dropout

        self.projectors = ModalityProjectors(
            modality_dims, modality_types, d_model,
            custom_projectors=custom_projectors,
        )

        self.mod_emb = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(1, d_model)) for m in modalities
        })
        for p in self.mod_emb.values():
            nn.init.normal_(p, 0.0, 0.02)

    def _tokens(
        self,
        modalities: Dict[str, Optional[torch.Tensor]],
        masks: Optional[Dict[str, Optional[torch.Tensor]]],
    ) -> torch.Tensor:
        """Return the [B, M, D] stack of available modality tokens."""
        projected = self.projectors(modalities, masks)
        filtered: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            if m in projected:
                filtered[m] = projected[m] + self.mod_emb[m]
        filtered = apply_modality_dropout(filtered, self.modality_dropout, self.training)
        if not filtered:
            raise ValueError("No modalities available for forward pass.")
        return torch.stack(list(filtered.values()), dim=1)  # [B, M, D]


# ---------------------------------------------------------------------------
# ABMIL — mean fusion over modality tokens
# ---------------------------------------------------------------------------

class ABMILFusionModel(_FusionBase):
    """Attention-based MIL fusion: modality tokens averaged, then predicted."""

    def __init__(
        self,
        modalities: List[str],
        modality_dims: Dict[str, int],
        modality_types: Dict[str, str],
        d_model: int = 256,
        output_dim: int = 1,
        modality_dropout: float = 0.0,
        fusion_mode: str = "mean",
        head_dropout: float = 0.0,
        custom_projectors: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self._build_common(
            modalities, modality_dims, modality_types,
            d_model, modality_dropout, custom_projectors,
        )
        self.head = PredictionHead(d_model, output_dim, dropout=head_dropout)

    def _fuse(
        self,
        modalities: Dict[str, Optional[torch.Tensor]],
        masks: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        tokens = self._tokens(modalities, masks)   # [B, M, D]
        return tokens.mean(dim=1)                  # [B, D]

    def forward(
        self,
        modalities: Dict[str, Optional[torch.Tensor]],
        masks: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        return self.head(self._fuse(modalities, masks))


# ---------------------------------------------------------------------------
# CLS-token transformer fusion
# ---------------------------------------------------------------------------

class ModalityFusionModel(_FusionBase):
    """Transformer encoder over modality tokens with a learned [CLS] token."""

    def __init__(
        self,
        modalities: List[str],
        modality_dims: Dict[str, int],
        modality_types: Dict[str, str],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
        modality_dropout: float = 0.0,
        head_dropout: float = 0.0,
        custom_projectors: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__()
        self._build_common(
            modalities, modality_dims, modality_types,
            d_model, modality_dropout, custom_projectors,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, 0.0, 0.02)
        self.fuser = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = PredictionHead(d_model, output_dim, dropout=head_dropout)

    def forward(
        self,
        modalities: Dict[str, Optional[torch.Tensor]],
        masks: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        tokens = self._tokens(modalities, masks)               # [B, M, D]
        bsz = tokens.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)               # [B, 1, D]
        x = torch.cat([cls, tokens], dim=1)                    # [B, M+1, D]
        x = self.fuser(x)
        z = x[:, 0]                                            # CLS output
        return self.head(z)


# ---------------------------------------------------------------------------
# Perceiver-style cross-attention fusion
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """One cross-attention layer (queries attend to context)."""

    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(q, context, context)
        q = self.norm1(q + self.drop(attn_out))
        q = self.norm2(q + self.drop(self.ff(q)))
        return q


class SelfAttentionBlock(nn.Module):
    """One self-attention layer (post-cross-attention refinement)."""

    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class CrossAttentionFusionModel(_FusionBase):
    """Learned queries cross-attend to modality tokens, then self-attend."""

    def __init__(
        self,
        modalities: List[str],
        modality_dims: Dict[str, int],
        modality_types: Dict[str, str],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        num_queries: int = 4,
        post_sab_layers: int = 1,
        dropout: float = 0.1,
        output_dim: int = 1,
        modality_dropout: float = 0.0,
        head_dropout: float = 0.0,
        custom_projectors: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__()
        self._build_common(
            modalities, modality_dims, modality_types,
            d_model, modality_dropout, custom_projectors,
        )

        ff_dim = d_model * 4
        self.queries = nn.Parameter(torch.zeros(num_queries, d_model))
        nn.init.normal_(self.queries, 0.0, 0.02)

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.self_attn_blocks = nn.ModuleList([
            SelfAttentionBlock(d_model, nhead, ff_dim, dropout)
            for _ in range(post_sab_layers)
        ])

        self.head = PredictionHead(d_model, output_dim, dropout=head_dropout)

    def forward(
        self,
        modalities: Dict[str, Optional[torch.Tensor]],
        masks: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        context = self._tokens(modalities, masks)                # [B, M, D]
        bsz = context.shape[0]
        q = self.queries.unsqueeze(0).expand(bsz, -1, -1)         # [B, Q, D]

        for blk in self.cross_attn_blocks:
            q = blk(q, context)
        for blk in self.self_attn_blocks:
            q = blk(q)

        z = q[:, 0]                                               # first query
        return self.head(z)
