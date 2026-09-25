"""mmfusion — multimodal fusion and MIL modelling for histopathology + radiology.

This package is a *behaviour-preserving* refactor of the original flat module
layout (``config.py``, ``data.py``, ``losses_metrics.py``, ``train.py``, …) plus
the pipeline logic that previously existed only inside the
``Pathology_only_TCGR_*.ipynb`` notebooks.

Design contract
---------------
No scientific calculation was changed. Every numeric path in this package is
covered by a parity test in ``tests/`` that asserts bit-identical output against
the frozen original implementation in ``legacy/_frozen/``. Known defects in the
original code (see ``REFACTOR_NUMERICS.md``) are reproduced exactly and marked
with ``BUG-PRESERVED`` comments rather than silently corrected — fixing them is
a separate, explicitly requested change.

Two model arms live here:

``mmfusion.training.mil``
    The gated-attention MIL pipeline extracted from the notebooks. This is what
    produced everything in ``Results/``.

``mmfusion.training.fusion``
    The general multimodal fusion framework (the ``mmfusion_flat`` code), which
    produced the W&B runs and the CHIMERA checkpoints.
"""

__version__ = "0.2.0"

from mmfusion import config, losses, metrics, paths, seeding

__all__ = ["config", "losses", "metrics", "paths", "seeding", "__version__"]
