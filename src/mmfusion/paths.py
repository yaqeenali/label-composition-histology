"""Path resolution.

Replaces the hard-coded ``C:\\Users\\styaaliii\\...`` literals that appeared in
every notebook *and* in the ``pt_path`` column of every split CSV, which
together made the project runnable on exactly one machine.

Nothing here changes a number: a rebased path must resolve to the same file. The
parity tests assert that :func:`rebase` is identity when no roots are configured,
and the MIL dataset defaults to using the stored path untouched.

Resolution order for a project root:

1. explicit argument
2. ``MMFUSION_ROOT`` environment variable
3. the ``paths.yaml`` / ``paths.json`` file next to the package's parent
4. the current working directory
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Optional, Union

__all__ = ["PathConfig", "project_root", "rebase", "resolve_case_path"]

_ENV_ROOT = "MMFUSION_ROOT"


def project_root(explicit: Optional[Union[str, Path]] = None) -> Path:
    """Return the project root directory."""
    if explicit is not None:
        return Path(explicit).resolve()
    env = os.environ.get(_ENV_ROOT)
    if env:
        return Path(env).resolve()
    # package lives at <root>/src/mmfusion/paths.py
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "data").exists() or (candidate / "configs").exists():
        return candidate
    return Path.cwd().resolve()


@dataclass
class PathConfig:
    """Roots used to rebase absolute paths recorded on another machine.

    ``rewrites`` maps a recorded prefix (as it appears in the CSV, Windows
    separators and all) to a local directory.
    """
    root: Optional[str] = None
    rewrites: Dict[str, str] = field(default_factory=dict)

    def resolve(self, recorded: str) -> str:
        return rebase(recorded, self.rewrites)


def _as_pure(path_str: str):
    """Parse a recorded path with the flavour it was written in."""
    if "\\" in path_str and not path_str.startswith("/"):
        return PureWindowsPath(path_str)
    return PurePosixPath(path_str)


def rebase(recorded: str, rewrites: Optional[Dict[str, str]] = None) -> str:
    """Map a recorded absolute path onto this machine.

    With no ``rewrites`` this is the identity function, which is what every
    legacy code path gets. When a prefix matches, the tail of the recorded path
    is re-rooted under the replacement, converting separators as needed.
    """
    if not rewrites:
        return recorded
    for prefix, replacement in rewrites.items():
        norm_recorded = recorded.replace("\\", "/")
        norm_prefix = prefix.replace("\\", "/").rstrip("/")
        if norm_recorded.startswith(norm_prefix):
            tail = norm_recorded[len(norm_prefix):].lstrip("/")
            return str(Path(replacement) / tail)
    return recorded


def resolve_case_path(
    source_dir: Union[str, Path],
    case_id: str,
    fold_name: Optional[str] = None,
    suffix: str = ".pt",
) -> Path:
    """``<source_dir>/[<fold_name>/]<case_id><suffix>``.

    Mirrors the lookup :class:`MultiModalDataset` performs, kept in one place so
    dimension inference and item loading cannot drift apart.
    """
    base = Path(source_dir)
    if fold_name is not None:
        base = base / fold_name
    return base / f"{case_id}{suffix}"
