"""Transfer-learning utilities.

Moved from the top-level ``transfer.py`` without modification: shape-matched
weight loading, modality-token seeding, freeze/unfreeze helpers and the
progressive-unfreezing scheduler. It has no dependency on the rest of the
package, so the body below is byte-identical to the original (asserted by
``tests/test_parity_transfer.py::test_source_is_unmodified``).
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    path: str,
    extra: Optional[dict] = None,
) -> None:
    """Save model weights (and optional metadata) to *path*."""
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = True,
    map_location: str = "cpu",
) -> dict:
    """Load weights from *path* into *model*.

    Returns the ``incompatible_keys`` named-tuple so callers can inspect
    which keys were missing / unexpected.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt.get("state_dict", ckpt)          # handle bare state_dict or wrapped dict
    result = model.load_state_dict(state, strict=strict)
    if not strict:
        if result.missing_keys:
            logger.info(
                f"  Missing keys (initialised randomly): {result.missing_keys[:10]}"
                + ("..." if len(result.missing_keys) > 10 else "")
            )
        if result.unexpected_keys:
            logger.info(
                f"  Unexpected keys (ignored): {result.unexpected_keys[:10]}"
                + ("..." if len(result.unexpected_keys) > 10 else "")
            )
    return result


# ---------------------------------------------------------------------------
# Shape-matched weight loading
# ---------------------------------------------------------------------------

def load_matching_weights(
    target_model: nn.Module,
    checkpoint_path: str,
    map_location: str = "cpu",
) -> Dict[str, List[str]]:
    """Load only the weights whose tensor shapes match exactly.

    Unlike ``strict=False`` (which copies everything and silently skips keys
    that don't exist in the target), this function also skips keys where the
    shape differs.  This is essential when the same architecture is used
    across datasets with different embedding dimensions – e.g. a WSI projector
    trained on 1024-dim patch features must not be loaded into a model expecting
    512-dim features even though the parameter *name* is the same.

    Rules
    -----
    - Key exists in source AND target AND shapes match → **copied**.
    - Key exists in source but NOT in target → **skipped (new arch)**.
    - Key exists in source AND target but shapes differ → **skipped (dim mismatch)**.
    - Key exists in target but NOT in source → **skipped (new layer)**.

    Returns
    -------
    A dict with lists ``"copied"``, ``"shape_mismatch"``, ``"not_in_target"``,
    ``"not_in_source"`` for full transparency.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    src_state: dict = ckpt.get("state_dict", ckpt)
    tgt_state: dict = target_model.state_dict()

    report: Dict[str, List[str]] = {
        "copied": [],
        "shape_mismatch": [],
        "not_in_target": [],
        "not_in_source": [],
    }

    new_state = {k: v.clone() for k, v in tgt_state.items()}  # start from target weights

    for key, src_tensor in src_state.items():
        if key not in tgt_state:
            report["not_in_target"].append(key)
            continue
        tgt_tensor = tgt_state[key]
        if src_tensor.shape != tgt_tensor.shape:
            report["shape_mismatch"].append(
                f"{key}: src={tuple(src_tensor.shape)} tgt={tuple(tgt_tensor.shape)}"
            )
            continue
        new_state[key] = src_tensor.clone()
        report["copied"].append(key)

    for key in tgt_state:
        if key not in src_state:
            report["not_in_source"].append(key)

    target_model.load_state_dict(new_state, strict=True)

    # --- summary log ---------------------------------------------------------
    n_copied = len(report["copied"])
    n_mismatch = len(report["shape_mismatch"])
    n_new_tgt = len(report["not_in_source"])
    n_new_src = len(report["not_in_target"])

    logger.info(
        f"load_matching_weights: {n_copied} copied, "
        f"{n_mismatch} shape-mismatched (kept random init), "
        f"{n_new_tgt} new in target (kept random init), "
        f"{n_new_src} only in source (ignored)."
    )
    if report["shape_mismatch"]:
        logger.info("  Shape mismatches (skipped):")
        for entry in report["shape_mismatch"]:
            logger.info(f"    {entry}")
    if report["not_in_source"]:
        logger.info(
            f"  New target layers (random init): "
            + ", ".join(report["not_in_source"][:8])
            + ("..." if len(report["not_in_source"]) > 8 else "")
        )
    return report


# ---------------------------------------------------------------------------
# Encoder-only transfer
# ---------------------------------------------------------------------------

def load_pretrained_encoder(
    target_model: nn.Module,
    checkpoint_path: str,
    map_location: str = "cpu",
    match_shapes: bool = True,
) -> Dict[str, List[str]]:
    """Transfer encoder weights from a pre-trained checkpoint into *target_model*.

    Parameters
    ----------
    match_shapes : bool (default True)
        When True (recommended) uses ``load_matching_weights`` so that only
        parameters with identical shapes are transferred.  This prevents
        silently copying a projector layer that was trained on a different
        embedding dimension (e.g. 1024-dim WSI features into a 512-dim model).
        When False falls back to ``strict=False`` behaviour (legacy).
    """
    logger.info(f"Loading pretrained encoder from: {checkpoint_path}")
    if match_shapes:
        return load_matching_weights(target_model, checkpoint_path, map_location)
    # legacy path
    result = load_checkpoint(target_model, checkpoint_path, strict=False, map_location=map_location)
    return {"copied": [], "shape_mismatch": [], "not_in_target": result.unexpected_keys,
            "not_in_source": result.missing_keys}


# ---------------------------------------------------------------------------
# Semantic modality-token seeding
# ---------------------------------------------------------------------------

def seed_modality_tokens(
    target_model: nn.Module,
    checkpoint_path: str,
    alias_map: Optional[Dict[str, str]] = None,
    map_location: str = "cpu",
) -> Dict[str, str]:
    """Copy pre-trained modality-token embeddings (``mod_emb``) into *target_model*.

    This is useful when the new dataset has **semantically equivalent** data
    under a different modality name, or when the input projector dimension
    changed (the token embedding itself is always transferable because its
    shape ``[1, d_model]`` does not depend on the input feature dimension).

    Parameters
    ----------
    target_model : nn.Module
        The new model whose ``mod_emb`` entries should be seeded.
    checkpoint_path : str
        Path to the pre-trained checkpoint.
    alias_map : dict, optional
        Mapping ``{new_name: pretrained_name}``.  For example
        ``{"clinical_v2": "clin"}`` seeds the ``clinical_v2`` token with the
        weights learned for ``clin`` in pre-training.
        If *None*, only same-name modalities are seeded (identical to what
        ``load_matching_weights`` already does for ``mod_emb``).
    map_location : str
        Passed to ``torch.load``.

    Returns
    -------
    dict
        ``{target_name: source_name}`` for each token that was seeded,
        ``{target_name: "not_found"}`` for modalities that couldn't be matched.

    Notes
    -----
    Only the ``mod_emb`` parameters are touched; input projectors and all
    other layers are left unchanged.  Call this *after*
    ``load_matching_weights`` / ``load_pretrained_encoder`` so that
    same-name modalities are not seeded twice.
    """
    alias_map = alias_map or {}

    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    src_state: dict = ckpt.get("state_dict", ckpt)

    # Collect all mod_emb entries from the checkpoint:  "mod_emb.clin" → tensor
    src_tokens: Dict[str, torch.Tensor] = {
        key[len("mod_emb."):]: tensor
        for key, tensor in src_state.items()
        if key.startswith("mod_emb.")
    }

    if not hasattr(target_model, "mod_emb"):
        logger.warning("target_model has no 'mod_emb' attribute – nothing seeded.")
        return {}

    report: Dict[str, str] = {}
    for tgt_name, param in target_model.mod_emb.items():  # type: ignore[union-attr]
        # Determine which source token to use
        src_name = alias_map.get(tgt_name, tgt_name)
        if src_name not in src_tokens:
            logger.info(f"  seed_modality_tokens: '{tgt_name}' → source '{src_name}' not found, kept random init.")
            report[tgt_name] = "not_found"
            continue
        src_tensor = src_tokens[src_name]
        if src_tensor.shape != param.shape:
            logger.warning(
                f"  seed_modality_tokens: '{tgt_name}' shape mismatch "
                f"src={tuple(src_tensor.shape)} tgt={tuple(param.shape)}, skipped."
            )
            report[tgt_name] = "shape_mismatch"
            continue
        with torch.no_grad():
            param.copy_(src_tensor)
        label = f"{src_name}" if src_name == tgt_name else f"{src_name} (alias)"
        logger.info(f"  seed_modality_tokens: '{tgt_name}' ← '{label}'")
        report[tgt_name] = src_name

    seeded = sum(1 for v in report.values() if v not in ("not_found", "shape_mismatch"))
    logger.info(f"seed_modality_tokens: {seeded}/{len(report)} token(s) seeded.")
    return report


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_by_prefix(model: nn.Module, prefixes: List[str]) -> List[str]:
    """Freeze all parameters whose names start with any of *prefixes*.

    Returns the list of frozen parameter names (useful for logging).
    """
    frozen: List[str] = []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad_(False)
            frozen.append(name)
    logger.info(f"Frozen {len(frozen)} parameters matching prefixes {prefixes}")
    return frozen


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad_(True)
    logger.info("All parameters unfrozen.")


def freeze_encoder(model: nn.Module) -> List[str]:
    """Convenience: freeze everything except the prediction head.

    Assumes the head is named ``head`` (as in PredictionHead).
    """
    head_prefix = "head."
    frozen: List[str] = []
    for name, param in model.named_parameters():
        if not name.startswith(head_prefix):
            param.requires_grad_(False)
            frozen.append(name)
    logger.info(f"Encoder frozen ({len(frozen)} params).  Head remains trainable.")
    return frozen


def print_trainable_params(model: nn.Module) -> None:
    """Print a summary of trainable vs frozen parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    print(f"Parameters: {total:,} total | {trainable:,} trainable | {frozen:,} frozen")


# ---------------------------------------------------------------------------
# Progressive unfreezing schedule
# ---------------------------------------------------------------------------

class ProgressiveUnfreezeScheduler:
    """Gradually unfreeze layers during training for fine-tuning.

    At each ``step_epoch`` the *next* group of parameter prefixes is unfrozen.
    Groups should be ordered from head → encoder (fine-tuning direction).

    Example
    -------
    scheduler = ProgressiveUnfreezeScheduler(
        model,
        groups=[["head"], ["cross_attn_blocks"], ["projectors"]],
        step_epochs=[0, 5, 15],
    )
    # In training loop:
    scheduler.step(epoch)
    """

    def __init__(
        self,
        model: nn.Module,
        groups: List[List[str]],
        step_epochs: List[int],
    ):
        if len(groups) != len(step_epochs):
            raise ValueError("groups and step_epochs must have the same length")
        self.model = model
        self.groups = groups
        self.step_epochs = step_epochs
        self._unfrozen_up_to = -1

        # Start with everything frozen
        for param in model.parameters():
            param.requires_grad_(False)

    def step(self, epoch: int) -> None:
        for i, trigger in enumerate(self.step_epochs):
            if epoch >= trigger and i > self._unfrozen_up_to:
                for name, param in self.model.named_parameters():
                    if any(name.startswith(p) for p in self.groups[i]):
                        param.requires_grad_(True)
                self._unfrozen_up_to = i
                logger.info(
                    f"Epoch {epoch}: unfreezing group {i} {self.groups[i]}"
                )
