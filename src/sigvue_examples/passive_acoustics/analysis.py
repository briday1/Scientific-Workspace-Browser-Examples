"""Waveform and spectral analysis for NOAA hydrophone clips."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from sigvue.plugin import ParameterContext

from ..plugins.sigmf import SigMFWindow


@dataclass(frozen=True)
class AcousticSettings:
    fft_size: int
    overlap_percent: int


@dataclass(frozen=True)
class AcousticProducts:
    window: SigMFWindow
    time_seconds: np.ndarray
    waveform: np.ndarray
    frequency_khz: np.ndarray
    spectrogram_dbfs: np.ndarray
    spectrogram_time_seconds: np.ndarray
    psd_dbfs_hz: np.ndarray


def configure(data: SigMFWindow, ui: ParameterContext) -> AcousticSettings:
    return AcousticSettings(
        fft_size=int(ui.select(
            "acoustic_fft_size", label="FFT size", default=2048,
            options=(512, 1024, 2048, 4096), group="Acoustic processing",
        )),
        overlap_percent=int(ui.select(
            "acoustic_overlap_percent", label="Overlap (%)", default=75,
            options=(0, 25, 50, 75), group="Acoustic processing",
        )),
    )


def process(data: SigMFWindow, settings: AcousticSettings | None) -> AcousticProducts:
    if settings is None:
        raise RuntimeError("Passive-acoustic analysis requires settings")
    samples = np.asarray(data.channel().real, dtype=np.float32)
    fft_size = min(settings.fft_size, samples.size)
    overlap = min(fft_size - 1, round(fft_size * settings.overlap_percent / 100))
    frequency, relative_time, spectrum = signal.spectrogram(
        samples,
        fs=data.recording.sample_rate,
        window="hann",
        nperseg=fft_size,
        noverlap=overlap,
        detrend=False,
        scaling="spectrum",
        mode="psd",
    )
    spectrogram_dbfs = 10 * np.log10(np.maximum(spectrum, 1e-12))
    psd_dbfs_hz = 10 * np.log10(np.maximum(np.mean(spectrum, axis=1), 1e-12))
    start = data.start_seconds
    return AcousticProducts(
        window=data,
        time_seconds=start + np.arange(samples.size) / data.recording.sample_rate,
        waveform=samples,
        frequency_khz=frequency / 1e3,
        spectrogram_dbfs=spectrogram_dbfs.T,
        spectrogram_time_seconds=start + relative_time,
        psd_dbfs_hz=psd_dbfs_hz,
    )


__all__ = ["AcousticProducts", "AcousticSettings", "configure", "process"]
