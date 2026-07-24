"""Typed values exchanged by the calibrated LFM radar pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sigvue.plugin import Annotation

from ..plugins.sigmf import SigMFRecording, load_sigmf_recording


@dataclass(frozen=True)
class CollectionMember:
    role: str
    channel: int
    metadata_path: Path
    data_path: Path
    duration: float


@dataclass(frozen=True)
class LfmCollection:
    sample_rate: float
    calibration_dbm: float
    adc_bits: int
    members: dict[str, tuple[CollectionMember, ...]]
    ota_prf_hz: float = 1_000.0
    ota_pulse_width_seconds: float = 50e-6
    collection_path: Path | None = None

    def sample_count(self, role: str) -> int:
        return min(
            _open_member_recording(member).sample_count
            for member in self.members[role]
        )

    def read(
        self,
        role: str,
        start: int = 0,
        count: int | None = None,
    ) -> np.ndarray:
        """Read channel-first native ADC counts for radar calibration."""
        available = self.sample_count(role)
        start = min(available, max(0, start))
        count = (
            available - start
            if count is None
            else min(max(0, count), available - start)
        )
        channels = [
            _open_member_recording(member).read(
                start,
                count,
                normalized=False,
            )[0]
            for member in self.members[role]
        ]
        return np.asarray(channels, dtype=np.complex64)


def _open_member_recording(member: CollectionMember) -> SigMFRecording:
    """Reload metadata so live-growing members expose their current length."""
    recording = load_sigmf_recording(member.metadata_path)
    if recording.data_path.resolve() != member.data_path.resolve():
        raise ValueError(
            f"{member.metadata_path.name} does not reference "
            f"{member.data_path.name}"
        )
    if recording.channel_count != 1:
        raise ValueError(
            f"{member.metadata_path.name} must contain one channel"
        )
    return recording


@dataclass(frozen=True)
class LfmInput:
    sample_rate: float
    calibration_dbm: float
    adc_bits: int
    pri_samples: int
    start_sample: int
    calibration_counts: np.ndarray
    noise_counts: np.ndarray
    ota_counts: np.ndarray
    annotations: tuple[Annotation, ...] = ()


@dataclass(frozen=True)
class Calibration:
    phase_offsets: np.ndarray
    volts_per_count: np.ndarray
    amplitude_corrections: np.ndarray
    reference_volts_per_count: float
    phase_reference_channel: int
    amplitude_reference_channel: int
    amplitude_reference_label: str
    noise_power_dbm: np.ndarray
    noise_psd_dbm_hz: np.ndarray
    noise_figure_db: np.ndarray
    full_scale_dbm: np.ndarray


@dataclass(frozen=True)
class Products:
    fast_time_us: np.ndarray
    slow_time_s: np.ndarray
    slow_time_edges_s: np.ndarray
    frequencies_hz: np.ndarray
    time_mean_dbm: np.ndarray
    time_max_dbm: np.ndarray
    time_waterfall_dbm: np.ndarray
    psd_mean_dbm_hz: np.ndarray
    psd_max_dbm_hz: np.ndarray
    psd_waterfall_dbm_hz: np.ndarray


@dataclass(frozen=True)
class LfmSettings:
    adc_bits: int
    phase_reference: str
    amplitude_reference: str
    reference_noise_psd_dbm_hz: float


@dataclass(frozen=True)
class LfmAnalysisProducts:
    data: LfmInput
    settings: LfmSettings
    calibration: Calibration
    signal: Products
    calibrated_tone: np.ndarray
    calibrated_noise: np.ndarray
    phase_rows: list[dict[str, object]]
    amplitude_rows: list[dict[str, object]]
    amplitude_summary: str
    noise_rows: list[dict[str, object]]
