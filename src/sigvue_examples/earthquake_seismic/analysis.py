"""Time and frequency analysis of one seismic window."""

from dataclasses import dataclass

import numpy as np
from scipy import signal
from sigvue.plugin import ParameterContext

from ..plugins.sigmf import SigMFWindow


@dataclass(frozen=True)
class SeismicProducts:
    window: SigMFWindow
    time_seconds: np.ndarray
    velocity_um_s: np.ndarray
    frequency_hz: np.ndarray
    spectrogram_db: np.ndarray
    spectrogram_time_seconds: np.ndarray
    psd_db: np.ndarray


def configure(data: SigMFWindow, ui: ParameterContext) -> int:
    return int(ui.select(
        "seismic_fft_size", label="FFT size", default=512,
        options=(128, 256, 512, 1024), group="Seismic processing",
    ))


def process(data: SigMFWindow, fft_size: int | None) -> SeismicProducts:
    if fft_size is None:
        raise RuntimeError("Seismic analysis requires an FFT size")
    velocity = np.asarray(data.channel().real, dtype=np.float64)
    size = min(fft_size, velocity.size)
    frequency, time, spectrum = signal.spectrogram(
        velocity,
        fs=data.recording.sample_rate,
        nperseg=size,
        noverlap=size * 3 // 4,
        window="hann",
        scaling="density",
        mode="psd",
    )
    db = 10 * np.log10(np.maximum(spectrum, 1e-30))
    start = data.start_seconds
    return SeismicProducts(
        window=data,
        time_seconds=start + np.arange(velocity.size) / data.recording.sample_rate,
        velocity_um_s=velocity * 1e6,
        frequency_hz=frequency,
        spectrogram_db=db.T,
        spectrogram_time_seconds=start + time,
        psd_db=10 * np.log10(np.maximum(np.mean(spectrum, axis=1), 1e-30)),
    )
