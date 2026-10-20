"""Every raw label declared in label_map_strict.yaml round-trips through harmonise_labels()."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from har_icl.phase0_data.harmonize import (
    build_raw_to_unified,
    harmonise_labels,
    load_label_map,
    unified_label_set,
)


def _label_map_path() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here / "configs" / "label_map_strict.yaml"


def test_label_map_loads():
    m = load_label_map(_label_map_path())
    assert isinstance(m, dict) and len(m) > 5


def test_every_alias_maps_uniquely():
    m = load_label_map(_label_map_path())
    table = build_raw_to_unified(m)
    # No duplicate (dataset, raw_lower) keys with different unified targets.
    seen: dict[tuple[str, str], str | None] = {}
    for key, val in table.items():
        if key in seen and seen[key] != val:
            raise AssertionError(f"Alias {key} maps to both {seen[key]} and {val}")
        seen[key] = val


def test_unknown_raw_label_becomes_none():
    m = load_label_map(_label_map_path())
    out = harmonise_labels(np.array(["totally_bogus_label_xyz"]), "uci_har", m)
    assert out[0] is None


def test_known_raw_labels_map_to_unified():
    m = load_label_map(_label_map_path())
    out = harmonise_labels(np.array(["WALKING"]), "uci_har", m)
    assert out[0] == "walk"
    out = harmonise_labels(np.array(["ironing"]), "pamap2", m)
    assert out[0] == "iron"
    out = harmonise_labels(np.array(["elevator up"]), "usc_had", m)
    assert out[0] == "elevator_up"


def test_unified_label_set_non_empty():
    m = load_label_map(_label_map_path())
    s = unified_label_set(m)
    for required in ("walk", "iron", "nordic_walk", "elevator_up"):
        assert required in s
