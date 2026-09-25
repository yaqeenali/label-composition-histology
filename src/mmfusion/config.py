"""Configuration objects for both model arms.

The six original dataclasses (``ModalityConfig`` … ``WandbConfig``) are carried
over field-for-field, including every default, so existing callers and saved
``config.json`` records stay valid.

Three additions describe what used to be hard-coded inside the notebooks:

``MILModelConfig``  the ``GatedABMIL`` constructor arguments
``MILTrainConfig``  optimiser, epochs, loss and model-selection rule
``AssayConfig``     one assay end to end (columns, label mapping, the above)

These carry the notebook values as defaults *per preset*, not as new defaults —
see ``configs/assays/*.yaml``. Nothing here changes a computation; it moves
constants out of code and into data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "ModalityConfig",
    "TaskConfig",
    "ModelConfig",
    "TrainConfig",
    "TransferConfig",
    "WandbConfig",
    "MILModelConfig",
    "MILTrainConfig",
    "AssayConfig",
    "load_config",
    "dump_config",
]


# ===========================================================================
# Fusion-arm configs (unchanged from config.py)
# ===========================================================================

@dataclass
class ModalityConfig:
    """One input modality.

    ``pt_dir``  : directory of ``{case_id}.pt`` FloatTensors.
                  [N, D] -> bag (attention pooled); [D] -> vector (projected).
    ``tabular`` : columns of a shared CSV. ``feature_cols`` selects them; if it
                  is empty, *all numeric columns except* ``case_col`` are used.

    .. warning::
       The empty-``feature_cols`` default resolves to the target columns when
       run against the project's split CSVs. See ``REFACTOR_NUMERICS.md`` entry
       N-08 and :func:`mmfusion.data.multimodal.MultiModalDataset`.
    """
    name: str
    modality_type: str = "vector"            # 'bag' | 'vector'
    source_type: str = "pt_dir"              # 'pt_dir' | 'tabular'
    source_dir: Optional[str] = None
    source_csv: Optional[str] = None
    feature_cols: List[str] = field(default_factory=list)
    dim: Optional[int] = None                # inferred at runtime if None


@dataclass
class TaskConfig:
    """What to predict and how to score it.

    survival       -> Cox-PH loss, concordance index
    regression     -> MSE/MAE loss, R2 / Pearson / MAE
    classification -> BCE (binary) or CE (multi-class), AUROC / accuracy / F1
    """
    task_type: str = "survival"

    # survival
    time_col: str = "survival_months"
    censor_col: str = "censorship"           # 1 = censored (no event), 0 = event

    # regression / classification
    target_cols: List[str] = field(default_factory=lambda: ["target"])
    num_classes: int = 2

    # 'auto' -> inferred from task_type (cox_ph / mse / bce-or-ce)
    loss_fn: str = "auto"

    case_col: str = "case_id"


@dataclass
class ModelConfig:
    """Fusion architecture hyperparameters."""
    model_type: str = "crossattentionfusion"
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 2
    num_queries: int = 4
    post_sab_layers: int = 1
    dropout: float = 0.1
    modality_dropout: float = 0.1
    abmil_fusion_mode: str = "modality_instances"
    output_dim: Optional[int] = None         # inferred from the task if None


@dataclass
class TrainConfig:
    """Fusion training-loop hyperparameters."""
    max_epochs: int = 50
    lr: float = 1e-4
    wd: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 8
    num_workers: int = 0

    ema_eval: bool = False
    ema_decay: float = 0.999

    early_stopping: bool = False
    es_patience: int = 15
    es_metric: str = "auto"                  # 'auto' -> per-task default
    es_split: str = "val"
    es_higher_is_better: bool = True


@dataclass
class TransferConfig:
    """Pretrained-encoder loading."""
    enabled: bool = False
    checkpoint_path: Optional[str] = None
    freeze_prefixes: List[str] = field(default_factory=list)


@dataclass
class WandbConfig:
    """Experiment tracking."""
    enabled: bool = False
    project: str = "mmfusion"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_system: bool = True
    log_gradients: bool = False
    log_test_every_epoch: bool = True
    """Whether the test split is evaluated and logged each epoch.

    ``True`` reproduces the original loop, which is how the existing W&B history
    was produced, and is the default so that no existing configuration changes
    behaviour.

    .. warning::
       Setting this to ``False`` **changes trained weights and every downstream
       number.** It is not a logging-only switch. ``DataLoader.__iter__`` draws
       one value from the global torch RNG every time it is called, regardless
       of ``shuffle``, so removing an evaluation pass removes a draw and shifts
       the entire RNG stream from that point on.

       Use it for *new* runs where the test split should stay unseen during
       development; do not use it expecting to reproduce an existing run.
       Measured and asserted in
       ``tests/test_parity_training.py::test_test_logging_toggle_changes_results``.
       See ``REFACTOR_NUMERICS.md`` entry N-09.
    """


# ===========================================================================
# MIL-arm configs (new — extracted from notebook literals)
# ===========================================================================

@dataclass
class MILModelConfig:
    """``GatedABMIL`` constructor arguments."""
    input_dim: int = 1024
    hidden_dim: int = 1024
    dropout: float = 0.25


@dataclass
class MILTrainConfig:
    """MIL training loop settings, as the notebooks set them.

    ``selection_metric`` records which of the two divergent model-selection
    rules an assay used:

    ``val_loss``      keep the epoch with the lowest validation loss, using
                      ``<=`` so later ties win (GHI).
    ``val_pearson``   keep the epoch with the highest validation Pearson r,
                      using ``>`` so earlier ties win (all other assays).

    Both are reproduced exactly, tie-breaking included.
    """
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 1
    shuffle_train: bool = False
    num_patches: Optional[int] = None        # None = use every patch
    loss_type: str = "huber"                 # 'huber' | 'quantile' | 'weighted_huber'
    quantile_tau: float = 0.5
    weight_bins: int = 10
    selection_metric: str = "val_pearson"    # 'val_pearson' | 'val_loss'


@dataclass
class AssayConfig:
    """One assay, end to end."""
    name: str
    target_col: str
    group_col: Optional[str] = None          # source column for the binary label
    binary_target_col: Optional[str] = None  # derived column name
    label_map: Dict[str, int] = field(default_factory=dict)
    binary_class: bool = False               # legacy threshold-at-0.5 path
    splits_dir: str = ""
    binary_splits_dir: str = ""
    """Where the derived ``*_bin.csv`` live. Empty means ``splits_dir``.

    ``mmfusion-binarise`` writes out of place by default so ``data/`` stays
    read-only; set this to that directory so training reads what was just
    derived rather than whatever ``*_bin.csv`` happens to sit beside the splits.
    """
    split_suffix: str = "_bin"               # reads train{suffix}.csv etc.
    results_dir: str = ""
    model: MILModelConfig = field(default_factory=MILModelConfig)
    train: MILTrainConfig = field(default_factory=MILTrainConfig)


# ===========================================================================
# Serialisation
# ===========================================================================

_MIL_NESTED = {"model": MILModelConfig, "train": MILTrainConfig}


def _from_dict(cls, data: Dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys and nesting MIL configs."""
    known = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            continue
        if cls is AssayConfig and key in _MIL_NESTED and isinstance(value, dict):
            kwargs[key] = _MIL_NESTED[key](**value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path, cls=AssayConfig):
    """Load a YAML or JSON config file into *cls*.

    YAML is used when PyYAML is installed and the file has a YAML extension;
    otherwise the file is parsed as JSON. Keeping JSON working matters because
    ``run_fold`` already writes ``config.json`` records next to its results.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "PyYAML is required to read .yaml configs; "
                "install it or use a .json config instead."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return _from_dict(cls, data)


def dump_config(cfg, path) -> None:
    """Write a dataclass config to YAML or JSON, chosen by file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
