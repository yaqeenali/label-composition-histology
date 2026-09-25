"""Random-state control.

NUMERICAL CONTRACT
------------------
``set_seed`` is byte-for-byte the same routine that appeared (duplicated) in
``run_training.py`` and in every pathology notebook. The order and count of RNG
draws downstream of it must not change, so this module deliberately does *not*
introduce per-fold reseeding, generator objects, or ``torch.use_deterministic_
algorithms`` — all of which would alter results. See ``REFACTOR_NUMERICS.md``
entry N-01.

``seeded_rng`` is new and is *not* used by any legacy code path. It exists so
that new code (and the eventual fix for the unseeded bootstrap/permutation
routines) has somewhere to get an explicit generator without disturbing the
global state the legacy paths depend on.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

DEFAULT_SEED = 42
"""Seed used by ``make_splits`` and by every notebook pipeline.

``run_training.py`` called ``set_seed(242)`` at its entry point instead. That
discrepancy is preserved in :mod:`mmfusion.cli.train_fusion`; see
``REFACTOR_NUMERICS.md`` entry N-02.
"""


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and torch, and pin cuDNN to deterministic kernels.

    Verbatim reproduction of the original helper, including the fact that it
    seeds the *global* generators and is called once per session rather than
    once per fold.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seeded_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return an explicit NumPy generator.

    NOT used by any legacy path. Provided for new code so that resampling
    routines can be made reproducible without touching the global RNG stream
    that the legacy training loops consume.
    """
    return np.random.default_rng(seed)


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve a torch device using the original ``cuda if available else cpu`` rule."""
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
