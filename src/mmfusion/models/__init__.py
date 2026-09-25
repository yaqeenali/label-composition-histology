"""Model factory and public model surface."""
from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn

from mmfusion.config import ModelConfig, TaskConfig
from mmfusion.models.common import (
    AttentionPooling,
    ModalityProjectors,
    PredictionHead,
    apply_modality_dropout,
)
from mmfusion.models.fusion import (
    ABMILFusionModel,
    CrossAttentionBlock,
    CrossAttentionFusionModel,
    ModalityFusionModel,
    SelfAttentionBlock,
)
from mmfusion.models.gated_abmil import GatedABMIL

__all__ = [
    "ABMILFusionModel",
    "AttentionPooling",
    "CrossAttentionBlock",
    "CrossAttentionFusionModel",
    "GatedABMIL",
    "ModalityFusionModel",
    "ModalityProjectors",
    "PredictionHead",
    "SelfAttentionBlock",
    "apply_modality_dropout",
    "build_model",
]


def _resolve_output_dim(model_cfg: ModelConfig, task_cfg: TaskConfig) -> int:
    """Output width: 1 for survival/scalar regression, n_targets, or n_classes."""
    if model_cfg.output_dim is not None:
        return model_cfg.output_dim
    tt = task_cfg.task_type
    if tt == "classification":
        # Binary: single logit (bce); multi-class: n logits (ce)
        return 1 if task_cfg.num_classes <= 2 else task_cfg.num_classes
    if tt == "regression":
        return len(task_cfg.target_cols)
    return 1  # survival -> scalar log-hazard


def build_model(
    model_cfg: ModelConfig,
    task_cfg: TaskConfig,
    modality_dims: Dict[str, int],
    modality_types: Dict[str, str],
    custom_projectors: Optional[Dict[str, nn.Module]] = None,
) -> nn.Module:
    """Construct an untrained fusion model from config objects.

    ``custom_projectors`` overrides the default ``nn.Linear`` projector for the
    named modalities, e.g. to inject a ``ClinicalEmbedder`` for ``clin``.
    """
    modalities = list(modality_dims.keys())
    output_dim = _resolve_output_dim(model_cfg, task_cfg)

    common = dict(
        modalities=modalities,
        modality_dims=modality_dims,
        modality_types=modality_types,
        d_model=model_cfg.d_model,
        output_dim=output_dim,
        modality_dropout=model_cfg.modality_dropout,
        custom_projectors=custom_projectors,
    )

    mt = model_cfg.model_type.lower()
    if mt == "abmil":
        return ABMILFusionModel(**common)

    attn_common = dict(
        **common,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
    )
    if mt == "modalityfusion":
        return ModalityFusionModel(**attn_common)

    if mt == "crossattentionfusion":
        return CrossAttentionFusionModel(
            **attn_common,
            num_queries=model_cfg.num_queries,
            post_sab_layers=model_cfg.post_sab_layers,
        )

    raise ValueError(
        f"Unknown model_type: '{model_cfg.model_type}'. "
        "Choose from: abmil, modalityfusion, crossattentionfusion."
    )
