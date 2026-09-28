"""Tests for the dependency-free WFDB reader (Phase E4 PTB-XL signal I/O).

Synthetic records are built byte-for-byte here rather than mocked, so the tests
exercise the same parsing and decoding path the real PTB-XL files take. One
test additionally reads a genuine PTB-XL record when the sample has been
downloaded, and is skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.dataset.wfdb_reader import (
    EcgSignal,
    WfdbError,
    available_records,
    read_header,
    read_record,
)

_REAL_ROOT = Path(__file__).resolve().parents[1] / "backend" / "artifacts" / "dataset" / "ptbxl"
_REAL_STEM = _REAL_ROOT / "records100" / "00000" / "00001_lr"
_HAVE_REAL = _REAL_STEM.with_suffix(".hea").is_file() and _REAL_STEM.with_suffix(".dat").is_file()


def _checksum(column: np.ndarray) -> int:
    return int(np.sum(column.astype(np.int64)) & 0xFFFF)


def _write_record(
    directory: Path,
    name: str,
    raw: np.ndarray,
    fs: float = 100.0,
    gain: float = 1000.0,
    baseline: int = 0,
    fmt: str = "16",
    with_checksums: bool = True,
    n_samples_override: int | None = None,
    file_names: list[str] | None = None,
) -> Path:
    """Write a minimal single-file WFDB record and return its stem path."""
    n_samples, n_signals = raw.shape
    declared = n_samples if n_samples_override is None else n_samples_override
    lines = [f"{name} {n_signals} {fs:g} {declared}"]
    for index in range(n_signals):
        data_file = file_names[index] if file_names else f"{name}.dat"
        checksum = _checksum(raw[:, index]) if with_checksums else 0
        lines.append(
            f"{data_file} {fmt} {gain:g}({baseline})/mV 16 0 "
            f"{int(raw[0, index])} {checksum} 0 LEAD{index}"
        )
    (directory / f"{name}.hea").write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw.astype("<i2").tofile(directory / f"{name}.dat")
    return directory / name


def test_read_record_round_trips_physical_units(tmp_path: Path) -> None:
    raw = np.array([[100, -200], [300, -400], [0, 500]], dtype=np.int16)
    stem = _write_record(tmp_path, "rt", raw, gain=1000.0, baseline=0)

    signal = read_record(stem)

    assert signal.samples.shape == (3, 2)
    np.testing.assert_allclose(signal.samples[:, 0], [0.1, 0.3, 0.0])
    np.testing.assert_allclose(signal.samples[:, 1], [-0.2, -0.4, 0.5])
    assert signal.sampling_frequency == 100.0
    assert signal.lead_names == ("LEAD0", "LEAD1")
    assert signal.units == ("mV", "mV")
    assert signal.checksums_ok is True


def test_baseline_offset_is_subtracted(tmp_path: Path) -> None:
    raw = np.array([[1100], [1000], [900]], dtype=np.int16)
    stem = _write_record(tmp_path, "bl", raw, gain=100.0, baseline=1000)

    signal = read_record(stem)

    np.testing.assert_allclose(signal.samples[:, 0], [1.0, 0.0, -1.0])


def test_header_reports_geometry_and_duration(tmp_path: Path) -> None:
    raw = np.zeros((250, 3), dtype=np.int16)
    stem = _write_record(tmp_path, "geo", raw, fs=50.0)

    header = read_header(stem.with_suffix(".hea"))

    assert header.n_signals == 3
    assert header.sampling_frequency == 50.0
    assert header.n_samples == 250
    assert header.duration_seconds == pytest.approx(5.0)
    assert header.lead_names == ("LEAD0", "LEAD1", "LEAD2")


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    raw = np.array([[10], [20], [30]], dtype=np.int16)
    stem = _write_record(tmp_path, "cs", raw)
    # Corrupt the data only; the header keeps the checksum of the original samples.
    np.array([[10], [20], [999]], dtype=np.int16).astype("<i2").tofile(
        stem.with_suffix(".dat")
    )

    with pytest.raises(WfdbError, match="checksum mismatch"):
        read_record(stem)


def test_checksum_verification_can_be_disabled(tmp_path: Path) -> None:
    raw = np.array([[10], [20], [30]], dtype=np.int16)
    stem = _write_record(tmp_path, "nocs", raw)
    np.array([[10], [20], [999]], dtype=np.int16).astype("<i2").tofile(
        stem.with_suffix(".dat")
    )

    signal = read_record(stem, verify_checksums=False)

    assert signal.checksums_ok is None
    np.testing.assert_allclose(signal.samples[:, 0], [0.01, 0.02, 0.999])


def test_unsupported_format_raises_rather_than_misreading(tmp_path: Path) -> None:
    raw = np.zeros((4, 1), dtype=np.int16)
    stem = _write_record(tmp_path, "f212", raw, fmt="212")

    with pytest.raises(WfdbError, match="format 212"):
        read_record(stem)


@pytest.mark.parametrize(
    "fmt, message",
    [("16x2", "samples per frame"), ("16:5", "skew"), ("16+64", "byte offset")],
)
def test_unimplemented_header_features_raise(tmp_path: Path, fmt: str, message: str) -> None:
    raw = np.zeros((4, 1), dtype=np.int16)
    stem = _write_record(tmp_path, f"h{abs(hash(fmt)) % 1000}", raw, fmt=fmt)

    with pytest.raises(WfdbError, match=message):
        read_record(stem)


def test_truncated_data_file_raises(tmp_path: Path) -> None:
    raw = np.zeros((10, 2), dtype=np.int16)
    stem = _write_record(tmp_path, "trunc", raw, n_samples_override=50)

    with pytest.raises(WfdbError, match="truncated"):
        read_record(stem)


def test_multi_file_record_raises(tmp_path: Path) -> None:
    raw = np.zeros((4, 2), dtype=np.int16)
    stem = _write_record(tmp_path, "multi", raw, file_names=["a.dat", "b.dat"])

    with pytest.raises(WfdbError, match="single-file"):
        read_header(stem.with_suffix(".hea"))


def test_missing_header_and_missing_data_raise(tmp_path: Path) -> None:
    with pytest.raises(WfdbError, match="Header not found"):
        read_header(tmp_path / "nope.hea")

    raw = np.zeros((4, 1), dtype=np.int16)
    stem = _write_record(tmp_path, "nodat", raw)
    stem.with_suffix(".dat").unlink()
    with pytest.raises(WfdbError, match="Signal file not found"):
        read_record(stem)


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    raw = np.array([[5], [6]], dtype=np.int16)
    stem = _write_record(tmp_path, "cmt", raw)
    original = stem.with_suffix(".hea").read_text(encoding="utf-8")
    stem.with_suffix(".hea").write_text(
        "# a leading comment\n\n" + original + "\n# trailing\n", encoding="utf-8"
    )

    signal = read_record(stem)

    assert signal.n_samples == 2


def test_zero_gain_falls_back_to_wfdb_default(tmp_path: Path) -> None:
    raw = np.array([[200], [400]], dtype=np.int16)
    stem = _write_record(tmp_path, "zg", raw, gain=0.0)

    signal = read_record(stem)

    # WFDB's documented default gain for an uncalibrated signal is 200 adu/mV.
    np.testing.assert_allclose(signal.samples[:, 0], [1.0, 2.0])


def test_lead_lookup_is_case_insensitive_and_validated(tmp_path: Path) -> None:
    raw = np.array([[1, 2], [3, 4]], dtype=np.int16)
    stem = _write_record(tmp_path, "lead", raw)

    signal = read_record(stem)

    np.testing.assert_allclose(signal.lead("lead1"), signal.samples[:, 1])
    with pytest.raises(WfdbError, match="not in record"):
        signal.lead("V9")


def test_available_records_filters_to_complete_pairs(tmp_path: Path) -> None:
    raw = np.zeros((4, 1), dtype=np.int16)
    _write_record(tmp_path, "whole", raw)
    (tmp_path / "headeronly.hea").write_text("headeronly 1 100 4\nheaderonly.dat 16\n")

    present = available_records(tmp_path, ["whole", "headeronly", "absent", "  "])

    assert present == ["whole"]


def test_ecg_signal_geometry_helpers() -> None:
    signal = EcgSignal(
        samples=np.zeros((300, 2)),
        sampling_frequency=100.0,
        lead_names=("I", "II"),
        units=("mV", "mV"),
        record_name="synthetic",
    )

    assert signal.n_samples == 300
    assert signal.n_leads == 2
    assert signal.duration_seconds == pytest.approx(3.0)


@pytest.mark.skipif(not _HAVE_REAL, reason="PTB-XL signal sample not downloaded")
def test_real_ptbxl_record_decodes_with_verified_checksums() -> None:
    """Lock the decoded geometry of a genuine PTB-XL record.

    The published record 00001_lr is 12-lead, 100 Hz, 10 s. Passing WFDB
    checksums means the decode agrees with the values the publishers wrote.
    """
    signal = read_record(_REAL_STEM)

    assert signal.checksums_ok is True
    assert signal.samples.shape == (1000, 12)
    assert signal.sampling_frequency == 100.0
    assert signal.duration_seconds == pytest.approx(10.0)
    assert signal.lead_names == (
        "I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6",
    )
    assert set(signal.units) == {"mV"}
    # Physiological amplitude sanity: real ECG leads span well under +-10 mV.
    assert np.all(np.abs(signal.samples) < 10.0)
    assert float(np.ptp(signal.lead("II"))) > 0.1
