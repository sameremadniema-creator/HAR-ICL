"""pytest config: ensure HAR-CODEBOOK/ is on sys.path so `import har_icl.x` works"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent       # .../HAR-CODEBOOK/har_icl
PARENT = ROOT.parent                                 # .../HAR-CODEBOOK

# We expose the project both as `har_icl.xxx` (canonical) and as bare.
for p in (str(PARENT), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_collection_modifyitems(config, items):
    """Skip tests that depend on optional heavy deps when those deps are missing."""
    import importlib

    require = {
        "test_vqvae_shapes": ["torch"],
        "test_codebook_singletoken": ["torch", "transformers"],
        "test_prompt_budget": ["torch", "transformers"],
    }
    for item in items:
        mod = item.module.__name__.rsplit(".", 1)[-1]
        for need in require.get(mod, []):
            try:
                importlib.import_module(need)
            except ImportError:
                import pytest as _pytest

                item.add_marker(_pytest.mark.skip(reason=f"{need} not installed"))
                break
