"""Generic multimodal dataset for the fusion arm.

Handles three modality shapes:

* **bag**     ``.pt`` of shape [N, D] — attention pooled downstream
* **vector**  ``.pt`` of shape [D] or [1, D] — linearly projected
* **tabular** columns of a shared CSV

Carried over from ``data.py`` with the loading, padding and target logic
unchanged. Two things worth knowing, both preserved:

* A missing modality file yields ``None``, is zero-filled by the collate
  function, and produces only a warning — nothing marks it as missing to the
  model (``REFACTOR_NUMERICS.md`` N-12).
* A tabular modality with no ``feature_cols`` selects every numeric column
  except the case column, which on this project's split CSVs means the target
  columns (N-08). :class:`MultiModalDataset` now warns loudly when that happens;
  the selection itself is unchanged.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from mmfusion.config import ModalityConfig, TaskConfig
from mmfusion.paths import resolve_case_path

logger = logging.getLogger(__name__)

__all__ = [
    "MultiModalDataset",
    "make_collate_fn",
    "infer_dims_from_dataset",
]


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _safe_load_pt(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_float_tensor(x) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x.float()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiModalDataset(Dataset):
    """Rows of a split CSV joined to per-modality features.

    Parameters
    ----------
    split_csv : CSV with a case column and the target column(s)
    modality_configs : ordered modality specs; ``dim`` is filled in in place
    task_config : what to predict
    tabular_csv : shared CSV for modalities with ``source_type='tabular'``
    fold_name : when set, ``.pt`` files are looked up under
        ``<source_dir>/<fold_name>/`` — used when embeddings are refitted per
        fold, which is the case for the clinical and radiomics modalities
    """

    def __init__(
        self,
        split_csv: str,
        modality_configs: List[ModalityConfig],
        task_config: TaskConfig,
        tabular_csv: Optional[str] = None,
        fold_name: Optional[str] = None,
    ):
        self.task = task_config
        self.mod_cfgs = modality_configs
        self.case_col = task_config.case_col
        self.fold_name = fold_name

        self.df = (
            pd.read_csv(split_csv)
            .drop_duplicates(subset=[self.case_col])
            .reset_index(drop=True)
        )

        self._tabular_maps: Dict[str, Dict[str, torch.Tensor]] = {}
        self._tabular_dims: Dict[str, int] = {}
        for mc in modality_configs:
            if mc.source_type != "tabular":
                continue
            csv_path = mc.source_csv or tabular_csv
            if csv_path is None:
                raise ValueError(
                    f"Modality '{mc.name}' has source_type='tabular' but no "
                    "source_csv was given and no global tabular_csv was provided."
                )
            tab_df = pd.read_csv(csv_path)
            feature_cols = self._resolve_tabular_cols(tab_df, csv_path, mc)
            mc.dim = len(feature_cols)
            self._tabular_dims[mc.name] = len(feature_cols)
            mapping: Dict[str, torch.Tensor] = {}
            for _, row in tab_df.iterrows():
                vals = row[feature_cols].astype(float).fillna(0.0).values
                mapping[str(row[self.case_col])] = torch.tensor(vals, dtype=torch.float32)
            self._tabular_maps[mc.name] = mapping

        for mc in modality_configs:
            if mc.source_type == "pt_dir" and mc.dim is None:
                mc.dim = self._infer_pt_dim(mc)

    # ------------------------------------------------------------------

    def _resolve_tabular_cols(
        self,
        full_df: pd.DataFrame,
        csv_path: str,
        mc: ModalityConfig,
    ) -> List[str]:
        """Feature columns for a tabular modality.

        Identical selection to the original, which read the CSV twice (once for
        headers, once for dtypes); it is read once here and both checks run
        against that frame. The resulting column list is unchanged — asserted by
        ``tests/test_parity_data.py::test_tabular_column_resolution_parity``.
        """
        if mc.feature_cols:
            missing = [c for c in mc.feature_cols if c not in full_df.columns]
            if missing:
                raise ValueError(
                    f"Modality '{mc.name}': columns {missing} not found in {csv_path}"
                )
            return mc.feature_cols

        exclude = {self.case_col}
        cols = [
            c for c in full_df.select_dtypes(include="number").columns
            if c not in exclude
        ]
        leaked = [c for c in cols if c in set(self.task.target_cols)]
        if leaked:
            warnings.warn(
                f"Modality '{mc.name}' has no feature_cols, so it selected every "
                f"numeric column in {csv_path}. That includes the target "
                f"column(s) {leaked}, which feeds the label in as a feature. "
                "Set ModalityConfig.feature_cols explicitly.",
                stacklevel=2,
            )
        return cols

    def _infer_pt_dim(self, mc: ModalityConfig, max_scan: int = 50) -> Optional[int]:
        """Scan up to *max_scan* rows for the first readable file's feature width."""
        if mc.source_dir is None:
            return None
        scanned = 0
        for _, row in self.df.iterrows():
            path = resolve_case_path(mc.source_dir, row[self.case_col], self.fold_name)
            if path.exists():
                x = _to_float_tensor(_safe_load_pt(path))
                if x.dim() == 1:
                    return x.shape[0]
                elif x.dim() == 2:
                    return x.shape[-1]
                scanned += 1
                if scanned >= max_scan:
                    break
        return None

    # ------------------------------------------------------------------

    def _load_modality(self, mc: ModalityConfig, case_id: str):
        """FloatTensor for one modality, or None when unavailable."""
        if mc.source_type == "tabular":
            return self._tabular_maps.get(mc.name, {}).get(case_id, None)

        if mc.source_dir is None:
            return None
        path = resolve_case_path(mc.source_dir, case_id, self.fold_name)
        if not path.exists():
            return None

        x = _to_float_tensor(_safe_load_pt(path))

        if mc.modality_type == "bag":
            if x.dim() == 1:
                x = x.unsqueeze(0)
            elif x.dim() > 2:
                x = x.view(x.shape[0], -1)
        else:
            if x.dim() == 0:
                x = x.unsqueeze(0)
            elif x.dim() == 2:
                x = x.squeeze(0)
            elif x.dim() > 2:
                x = x.view(-1)
        return x

    def _load_target(self, row: pd.Series) -> Dict:
        tc = self.task
        if tc.task_type == "survival":
            time = float(row[tc.time_col])
            censor = float(row[tc.censor_col])   # 1 = censored -> event 0
            return {
                "time": torch.tensor(time, dtype=torch.float32),
                "event": torch.tensor(1.0 - censor, dtype=torch.float32),
            }
        elif tc.task_type == "regression":
            vals = torch.tensor(
                row[tc.target_cols].astype(float).fillna(0.0).values,
                dtype=torch.float32,
            )
            return {"target": vals}
        elif tc.task_type == "classification":
            if len(tc.target_cols) == 1:
                # NOTE: requires an already-integer-encoded label column. A string
                # label (e.g. Pam50.Call) raises here; there is no encoder.
                label = int(row[tc.target_cols[0]])
                return {"label": torch.tensor(label, dtype=torch.long)}
            label = torch.tensor(row[tc.target_cols].astype(int).values, dtype=torch.long)
            return {"label": label}
        raise ValueError(f"Unknown task_type: {tc.task_type}")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        case_id = str(row[self.case_col])

        sample: Dict = {"case_id": case_id}
        sample.update(self._load_target(row))

        for mc in self.mod_cfgs:
            sample[mc.name] = self._load_modality(mc, case_id)

        for mc in self.mod_cfgs:
            if sample.get(mc.name) is None:
                logger.warning("%s is None for case %s", mc.name, case_id)

        return sample

    @property
    def modality_dims(self) -> Dict[str, Optional[int]]:
        return {mc.name: mc.dim for mc in self.mod_cfgs}

    @property
    def modality_types(self) -> Dict[str, str]:
        return {mc.name: mc.modality_type for mc in self.mod_cfgs}


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _pad_bags(
    bags: List[Optional[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad [N_i, D] tensors to [B, max_N, D] with a validity mask.

    When *every* bag in the batch is missing this returns a [B, 1, 1] tensor,
    whose width will not match the projector. Preserved as-is.
    """
    valid = [b for b in bags if b is not None]
    bsz = len(bags)
    if not valid:
        return torch.zeros(bsz, 1, 1), torch.zeros(bsz, 1, dtype=torch.bool)
    d = valid[0].shape[-1]
    max_n = max(b.shape[0] for b in valid)
    tensor = torch.zeros(bsz, max_n, d, dtype=torch.float32)
    mask = torch.zeros(bsz, max_n, dtype=torch.bool)
    for i, b in enumerate(bags):
        if b is None:
            continue
        n = b.shape[0]
        tensor[i, :n] = b
        mask[i, :n] = True
    return tensor, mask


def _stack_vectors(
    vecs: List[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    """Stack [D] tensors to [B, D], zero-filling missing entries."""
    valid = [v for v in vecs if v is not None]
    if not valid:
        return None
    d = valid[0].numel()
    bsz = len(vecs)
    out = torch.zeros(bsz, d, dtype=torch.float32)
    for i, v in enumerate(vecs):
        if v is not None:
            out[i] = v.view(-1)
    return out


def make_collate_fn(modality_configs: List[ModalityConfig], task_config: TaskConfig):
    """Return a ``collate_fn`` bound to this modality/task configuration."""

    def collate_fn(batch: List[Dict]) -> Dict:
        out: Dict = {"case_id": [x["case_id"] for x in batch]}

        tc = task_config
        if tc.task_type == "survival":
            out["time"] = torch.stack([x["time"] for x in batch])
            out["event"] = torch.stack([x["event"] for x in batch])
        elif tc.task_type == "regression":
            out["target"] = torch.stack([x["target"] for x in batch])
        elif tc.task_type == "classification":
            out["label"] = torch.stack([x["label"] for x in batch])

        for mc in modality_configs:
            raw = [x[mc.name] for x in batch]
            if mc.modality_type == "bag":
                tensor, mask = _pad_bags(raw)
                out[mc.name] = tensor
                out[f"{mc.name}_mask"] = mask
            else:
                out[mc.name] = _stack_vectors(raw)

        return out

    return collate_fn


def infer_dims_from_dataset(ds: MultiModalDataset, n_scan: int = 50) -> Dict[str, int]:
    """Fill in any modality dims still unknown after construction."""
    dims: Dict[str, int] = {}
    for mc in ds.mod_cfgs:
        if mc.dim is not None:
            dims[mc.name] = mc.dim
            continue
        for i in range(min(len(ds), n_scan)):
            val = ds[i].get(mc.name)
            if val is not None:
                dims[mc.name] = val.shape[-1] if mc.modality_type == "bag" else int(val.numel())
                mc.dim = dims[mc.name]
                break
        if mc.name not in dims:
            raise RuntimeError(
                f"Could not infer dim for modality '{mc.name}'. "
                "No samples found with this modality present."
            )
    return dims
