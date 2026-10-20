"""Phase 0 — load, harmonise, split, and persist (N, 128, 6) windows."""
from .splits import (  # noqa: F401
    make_subject_split,
    make_unseen_split,
    write_splits_from_arrays,
    load_splits,
)
