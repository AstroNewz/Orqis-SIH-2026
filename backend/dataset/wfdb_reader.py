"""Minimal, dependency-free WFDB reader for the PTB-XL signal files.

Why hand-rolled instead of ``pip install wfdb``: the reference ``wfdb`` package
pulls in a wide dependency tree, and this environment's link to PyPI is slow and
intermittent (see ``docs/PHASE_E4_DATA_EXPANSION.md`` §7). PTB-XL's records are a
single, fully specified WFDB variant — one ``.dat`` per record, signal format 16,
no skew, no byte offset, one sample per frame — so a focused reader is ~150 lines
of numpy and removes an install-time failure mode from the pipeline entirely.

Deliberately strict: any header feature this reader does not implement (a format
other than 16, a nonzero skew or byte offset, multiple samples per frame, or a
multi-file record) raises :class:`WfdbError` rather than being ignored. Silently
mis-reading a signal would poison every downstream feature, so an unsupported
header must fail loudly.

The WFDB per-signal checksum (the low 16 bits of the sum of that signal's
samples) is carried in the header, so :func:`read_record` can verify the decoded
samples against it — an integrity check on the signal data itself, independent of
the SHA-256 check on the file bytes.

Reference: WFDB header/signal specification, https://physionet.org/physiotools/wag/header-5.htm
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

SUPPORTED_FORMAT = 16
"""Signal format this reader implements: 16-bit little-endian two's complement."""


class WfdbError(RuntimeError):
    """Raised when a WFDB record is missing, malformed, or uses unsupported features."""


# ``<format>[x<spf>][:<skew>][+<byte_offset>]`` -- we accept only a bare format.
_FORMAT_RE = re.compile(
    r"^(?P<fmt>\d+)"
    r"(?:x(?P<spf>\d+))?"
    r"(?::(?P<skew>\d+))?"
    r"(?:\+(?P<offset>\d+))?$"
)

# ``<gain>[(<baseline>)][/<units>]`` -- gain may be absent/zero (meaning uncalibrated).
_GAIN_RE = re.compile(
    r"^(?P<gain>[-+]?[\d.eE+-]*)"
    r"(?:\((?P<baseline>[-+]?\d+)\))?"
    r"(?:/(?P<units>\S+))?$"
)

_DEFAULT_GAIN = 200.0
"""WFDB's documented default when a header records gain 0 (uncalibrated)."""


@dataclass(frozen=True)
class SignalSpec:
    """One signal-specification line from a ``.hea`` header."""

    file_name: str
    fmt: int
    gain: float
    baseline: int
    units: str
    adc_resolution: int
    adc_zero: int
    initial_value: Optional[int]
    checksum: Optional[int]
    block_size: int
    description: str


@dataclass(frozen=True)
class WfdbHeader:
    """Parsed ``.hea`` header for a single-file WFDB record."""

    record_name: str
    n_signals: int
    sampling_frequency: float
    n_samples: int
    signals: Tuple[SignalSpec, ...]

    @property
    def duration_seconds(self) -> float:
        if self.sampling_frequency <= 0:
            return 0.0
        return self.n_samples / self.sampling_frequency

    @property
    def lead_names(self) -> Tuple[str, ...]:
        return tuple(signal.description for signal in self.signals)


@dataclass(frozen=True)
class EcgSignal:
    """Decoded multi-lead signal in physical units.

    ``samples`` is ``(n_samples, n_leads)`` in the header's physical units (mV for
    PTB-XL). ``lead_names`` is aligned with the columns.
    """

    samples: np.ndarray
    sampling_frequency: float
    lead_names: Tuple[str, ...]
    units: Tuple[str, ...]
    record_name: str
    checksums_ok: Optional[bool] = None
    """True/False when verification ran, None when the header carried no checksums."""

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_leads(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        if self.sampling_frequency <= 0:
            return 0.0
        return self.n_samples / self.sampling_frequency

    def lead(self, name: str) -> np.ndarray:
        """Return one lead by name (case-insensitive)."""
        target = name.strip().upper()
        for index, lead_name in enumerate(self.lead_names):
            if lead_name.strip().upper() == target:
                return self.samples[:, index]
        raise WfdbError(f"Lead {name!r} not in record {self.record_name!r} {self.lead_names}")


def _parse_int(token: str, field: str) -> int:
    try:
        return int(token)
    except (TypeError, ValueError) as exc:
        raise WfdbError(f"Bad integer for {field}: {token!r}") from exc


def _parse_signal_line(line: str, index: int) -> SignalSpec:
    tokens = line.split()
    if len(tokens) < 2:
        raise WfdbError(f"Signal line {index} has too few fields: {line!r}")

    file_name = tokens[0]

    fmt_match = _FORMAT_RE.match(tokens[1])
    if not fmt_match:
        raise WfdbError(f"Unparseable format field on signal line {index}: {tokens[1]!r}")
    fmt = int(fmt_match.group("fmt"))
    spf = int(fmt_match.group("spf") or 1)
    skew = int(fmt_match.group("skew") or 0)
    byte_offset = int(fmt_match.group("offset") or 0)
    if fmt != SUPPORTED_FORMAT:
        raise WfdbError(
            f"Signal {index} uses format {fmt}; this reader implements only "
            f"format {SUPPORTED_FORMAT}."
        )
    if spf != 1:
        raise WfdbError(f"Signal {index} has {spf} samples per frame; only 1 is supported.")
    if skew:
        raise WfdbError(f"Signal {index} has nonzero skew ({skew}); not supported.")
    if byte_offset:
        raise WfdbError(f"Signal {index} has byte offset {byte_offset}; not supported.")

    gain = _DEFAULT_GAIN
    baseline = 0
    units = "mV"
    if len(tokens) > 2:
        gain_match = _GAIN_RE.match(tokens[2])
        if not gain_match:
            raise WfdbError(f"Unparseable gain field on signal line {index}: {tokens[2]!r}")
        raw_gain = (gain_match.group("gain") or "").strip()
        if raw_gain:
            try:
                parsed_gain = float(raw_gain)
            except ValueError as exc:
                raise WfdbError(f"Bad ADC gain on signal line {index}: {raw_gain!r}") from exc
            # A header gain of 0 means "uncalibrated"; WFDB's documented fallback is 200.
            gain = parsed_gain if parsed_gain != 0.0 else _DEFAULT_GAIN
        if gain_match.group("baseline") is not None:
            baseline = int(gain_match.group("baseline"))
        if gain_match.group("units"):
            units = gain_match.group("units")

    adc_resolution = _parse_int(tokens[3], "ADC resolution") if len(tokens) > 3 else 12
    adc_zero = _parse_int(tokens[4], "ADC zero") if len(tokens) > 4 else 0
    initial_value = _parse_int(tokens[5], "initial value") if len(tokens) > 5 else None
    checksum = _parse_int(tokens[6], "checksum") if len(tokens) > 6 else None
    block_size = _parse_int(tokens[7], "block size") if len(tokens) > 7 else 0
    description = " ".join(tokens[8:]) if len(tokens) > 8 else f"sig{index}"

    # WFDB treats an absent baseline as equal to ADC zero.
    if len(tokens) > 2 and _GAIN_RE.match(tokens[2]).group("baseline") is None:  # type: ignore[union-attr]
        baseline = adc_zero

    return SignalSpec(
        file_name=file_name,
        fmt=fmt,
        gain=gain,
        baseline=baseline,
        units=units,
        adc_resolution=adc_resolution,
        adc_zero=adc_zero,
        initial_value=initial_value,
        checksum=checksum,
        block_size=block_size,
        description=description,
    )


def read_header(header_path: Path) -> WfdbHeader:
    """Parse a WFDB ``.hea`` header.

    Raises:
        WfdbError: if the file is missing, empty, malformed, or describes a record
            this reader does not implement (multi-file, non-format-16, skewed).
    """
    header_path = Path(header_path)
    if not header_path.is_file():
        raise WfdbError(f"Header not found: {header_path}")

    lines: List[str] = []
    for raw in header_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue  # blank lines and comments carry no signal specification
        lines.append(stripped)
    if not lines:
        raise WfdbError(f"Header is empty: {header_path}")

    record_tokens = lines[0].split()
    if len(record_tokens) < 2:
        raise WfdbError(f"Malformed record line in {header_path}: {lines[0]!r}")

    record_name = record_tokens[0]
    if "/" in record_name:
        raise WfdbError(f"Multi-segment records are not supported: {lines[0]!r}")
    n_signals = _parse_int(record_tokens[1], "number of signals")
    if n_signals <= 0:
        raise WfdbError(f"Record declares {n_signals} signals: {header_path}")

    sampling_frequency = 250.0  # WFDB default when the field is absent
    if len(record_tokens) > 2:
        # The frequency field may be ``fs/counter`` -- only fs matters here.
        fs_token = record_tokens[2].split("/", 1)[0]
        try:
            sampling_frequency = float(fs_token)
        except ValueError as exc:
            raise WfdbError(f"Bad sampling frequency in {header_path}: {fs_token!r}") from exc
    n_samples = _parse_int(record_tokens[3], "number of samples") if len(record_tokens) > 3 else 0

    signal_lines = lines[1 : 1 + n_signals]
    if len(signal_lines) != n_signals:
        raise WfdbError(
            f"{header_path} declares {n_signals} signals but has {len(signal_lines)} "
            f"specification line(s)."
        )
    signals = tuple(_parse_signal_line(line, i) for i, line in enumerate(signal_lines))

    distinct_files = {signal.file_name for signal in signals}
    if len(distinct_files) != 1:
        raise WfdbError(
            f"{header_path} spreads signals over {len(distinct_files)} files; only "
            f"single-file records are supported."
        )

    return WfdbHeader(
        record_name=record_name,
        n_signals=n_signals,
        sampling_frequency=sampling_frequency,
        n_samples=n_samples,
        signals=signals,
    )


def _wfdb_checksum(column: np.ndarray) -> int:
    """WFDB signal checksum: low 16 bits of the sum of the raw samples."""
    return int(np.sum(column.astype(np.int64)) & 0xFFFF)


def read_record(record_stem: Path, verify_checksums: bool = True) -> EcgSignal:
    """Read a WFDB record (``<stem>.hea`` + ``<stem>.dat``) into physical units.

    Args:
        record_stem: Path to the record without extension, e.g.
            ``.../records100/00000/00001_lr``.
        verify_checksums: When True (default), compare the decoded samples against
            the per-signal checksums in the header and raise on a mismatch.

    Raises:
        WfdbError: on a missing/malformed record, a truncated ``.dat``, or a
            checksum mismatch.
    """
    stem = Path(record_stem)
    header = read_header(stem.with_suffix(".hea"))

    data_path = stem.parent / header.signals[0].file_name
    if not data_path.is_file():
        raise WfdbError(f"Signal file not found: {data_path}")

    raw = np.fromfile(data_path, dtype="<i2")
    n_signals = header.n_signals
    if raw.size % n_signals != 0:
        raise WfdbError(
            f"{data_path} holds {raw.size} samples, not divisible by {n_signals} signals."
        )
    available = raw.size // n_signals
    expected = header.n_samples or available
    if available < expected:
        raise WfdbError(
            f"{data_path} is truncated: header declares {expected} samples per signal, "
            f"file holds {available}."
        )

    # Format 16 is frame-interleaved: sample0_sig0..sample0_sigN, sample1_sig0, ...
    frames = raw[: expected * n_signals].reshape(expected, n_signals)

    checksums_ok: Optional[bool] = None
    if verify_checksums:
        declared = [signal.checksum for signal in header.signals]
        if any(value is not None for value in declared):
            mismatches: List[str] = []
            for index, signal in enumerate(header.signals):
                if signal.checksum is None:
                    continue
                actual = _wfdb_checksum(frames[:, index])
                if actual != (signal.checksum & 0xFFFF):
                    mismatches.append(
                        f"{signal.description}: header={signal.checksum & 0xFFFF} actual={actual}"
                    )
            checksums_ok = not mismatches
            if mismatches:
                raise WfdbError(
                    f"WFDB checksum mismatch in {stem.name}: " + "; ".join(mismatches[:4])
                )

    physical = np.empty(frames.shape, dtype=np.float64)
    for index, signal in enumerate(header.signals):
        physical[:, index] = (frames[:, index].astype(np.float64) - signal.baseline) / signal.gain

    return EcgSignal(
        samples=physical,
        sampling_frequency=header.sampling_frequency,
        lead_names=header.lead_names,
        units=tuple(signal.units for signal in header.signals),
        record_name=header.record_name,
        checksums_ok=checksums_ok,
    )


def available_records(root: Path, stems: Sequence[str]) -> List[str]:
    """Filter ``stems`` (dataset-relative, no extension) to those fully present.

    Used to run a pipeline against whatever subset of a large corpus has actually
    been downloaded, without pretending the rest exists.
    """
    root = Path(root)
    present: List[str] = []
    for stem in stems:
        cleaned = str(stem).strip()
        if not cleaned:
            continue
        base = root / cleaned
        if base.with_suffix(".hea").is_file() and base.with_suffix(".dat").is_file():
            present.append(cleaned)
    return present
