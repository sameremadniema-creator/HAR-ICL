"""Tests for the idempotent dataset download module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from har_icl.phase0_data.download import (
    DATA_URLS,
    download_dataset,
    is_present,
    status_report,
)


def _fab_uci_har(raw: Path) -> None:
    d = raw / "uci_har" / "UCI HAR Dataset" / "train" / "Inertial Signals"
    d.mkdir(parents=True, exist_ok=True)
    for axis in ("body_acc_x", "body_acc_y", "body_acc_z"):
        (d / f"{axis}_train.txt").write_text("0 0 0\n")


def _fab_pamap2(raw: Path) -> None:
    d = raw / "pamap2" / "PAMAP2_Dataset" / "Protocol"
    d.mkdir(parents=True, exist_ok=True)
    (d / "subject101.dat").write_text("0 1 0 0\n")


def _fab_mhealth(raw: Path) -> None:
    d = raw / "mhealth"
    d.mkdir(parents=True, exist_ok=True)
    (d / "mHealth_subject1.log").write_text("0 0\n")


def _fab_usc_had(raw: Path) -> None:
    d = raw / "usc_had" / "Subject1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a1t1.mat").write_bytes(b"\x00" * 16)


def test_is_present_returns_false_on_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        for name in DATA_URLS:
            present, reason = is_present(name, raw)
            assert present is False
            assert isinstance(reason, str) and reason


def test_is_present_returns_true_after_fab():
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        _fab_uci_har(raw); _fab_pamap2(raw); _fab_mhealth(raw); _fab_usc_had(raw)
        for name in DATA_URLS:
            present, reason = is_present(name, raw)
            assert present is True, f"{name}: {reason}"


def test_status_report_shape():
    with tempfile.TemporaryDirectory() as tmp:
        rows = status_report(Path(tmp))
        assert isinstance(rows, list) and len(rows) == len(DATA_URLS)
        for r in rows:
            assert {"name", "present", "path", "reason"}.issubset(r.keys())


def test_download_dataset_skips_when_present_no_network():
    """When the presence check passes, download_dataset must NOT touch the network."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        _fab_uci_har(raw)
        _fab_pamap2(raw)
        _fab_mhealth(raw)
        _fab_usc_had(raw)
        # Block all outbound HTTP — if a download is attempted, urlopen will fail.
        import urllib.request
        saved = urllib.request.urlopen

        def _blocked(*a, **kw):
            raise AssertionError("urlopen should not be called when dataset is present")

        urllib.request.urlopen = _blocked  # type: ignore[assignment]
        try:
            for name in DATA_URLS:
                result = download_dataset(name, raw, verbose=False)
                assert result["status"] == "present", f"{name}: {result}"
        finally:
            urllib.request.urlopen = saved  # type: ignore[assignment]


def test_force_attempts_download_even_when_present():
    """With force=True we re-invoke the fetcher."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        _fab_usc_had(raw)
        result = download_dataset("usc_had", raw, force=True, verbose=False)
        assert result["status"] == "manual_required"


def test_unknown_dataset_name_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(KeyError):
            is_present("not_a_dataset", Path(tmp))
        with pytest.raises(KeyError):
            download_dataset("not_a_dataset", Path(tmp), verbose=False)
