"""Gated attention-based MIL model for whole-slide bags.

Extracted verbatim from the ``Pathology_only_TCGR_*.ipynb`` notebooks, where it
was redefined identically in each. This is the model behind every number in
``Results/``.

The two notebook variants differ only in constructor arguments, which are now
supplied by :class:`mmfusion.config.MILModelConfig` rather than by editing a
default:

===============  ==========  ===========  =========
assay            input_dim   hidden_dim   dropout
===============  ==========  ===========  =========
GHI              1024        256          0.50
MammaPrint       1024        1024         0.25
ROR-P            1024        1024         0.25
ROR-S            1024        1024         0.25
===============  ==========  ===========  =========

Defaults below are the GHI notebook's, because that is what the bare
``GatedABMIL()`` call in that notebook produced. Callers must pass values
explicitly; the presets in ``configs/assays/`` do.
"""
from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

__all__ = ["GatedABMIL"]


class GatedABMIL(nn.Module):
    """Gated-attention MIL: encode patches, score them, attention-pool, predict.

    Operates on a single bag at a time (batch size 1). ``x`` is ``[N_patches,
    input_dim]`` and the attention softmax runs over dim 0 (patches).

    RNG-ORDER: the five ``nn.Linear`` layers are constructed in the order
    encoder, attn_V, attn_U, attn_w, head. Reordering them changes the
    initialisation of every layer.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.50,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.attn_V = nn.Linear(hidden_dim, hidden_dim)
        self.attn_U = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_emb: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : [N_patches, input_dim]
        return_emb : also return the pooled bag embedding (detached, on CPU)

        Returns
        -------
        (pred, attention) or (pred, attention, embedding)
        """
        H = self.encoder(x)
        A = torch.tanh(self.attn_V(H)) * torch.sigmoid(self.attn_U(H))
        A = self.attn_w(A)
        A = torch.softmax(A, dim=0)
        M = torch.sum(A * H, dim=0)
        pred = self.head(M)
        if return_emb:
            return pred.squeeze(), A.squeeze(), M.detach().cpu()
        return pred.squeeze(), A.squeeze()
