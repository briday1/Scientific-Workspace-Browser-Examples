"""Windowed spectrogram view of an LTE SigMF recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from workspace_browser.plugin import AnalysisWorkspace

from .sigmf import SigMFRecording, SigMFWindow, sigmf_source
from .style import style_figure


COLORMAPS = ("Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo", "Blues", "Greens", "Hot", "Jet")
FFT_SIZES = (512, 1024, 2048, 4096, 8192, 16384)
FFT_WINDOWS = ("Hann", "Hamming", "Blackman", "Rectangular")
OVERLAPS = (0, 25, 50, 75, 88)
DBFS_MIN = -90.0
DBFS_MAX = -20.0


class WindowedLteDelivery:
    """Select an interval over a sliding-median power overview."""

    def prepare(self, recording: SigMFRecording, ui) -> SigMFWindow:
        overview = ui.once(
            f"lte-median-power-overview:{recording.metadata_path}",
            lambda: _median_power_overview(recording),
        )
        start_seconds, end_seconds = ui.windowed(
            duration=recording.duration_seconds,
            default_window=min(0.005, recording.duration_seconds),
            minimum_window=min(0.001, recording.duration_seconds),
            step=min(0.001, recording.duration_seconds),
            overview=overview,
            overview_label="Sliding median power (dBFS)",
        )
        start = round(start_seconds * recording.sample_rate)
        count = max(1, round((end_seconds - start_seconds) * recording.sample_rate))
        return SigMFWindow(recording, start, recording.read(start, count))


def _median_power_overview(recording: SigMFRecording) -> np.ndarray:
    bin_count = min(400, recording.sample_count)
    edges = np.linspace(0, recording.sample_count, bin_count + 1, dtype=np.int64)
    power = np.empty(bin_count)
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        samples = recording.read(int(start), int(end - start))[0]
        power[index] = np.mean(np.abs(samples) ** 2)
    power_dbfs = 10 * np.log10(np.maximum(power, 1e-12))
    width = min(9, bin_count)
    if width % 2 == 0:
        width -= 1
    padded = np.pad(power_dbfs, width // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)


def analyze(data: SigMFWindow, ui) -> None:
    colormap = ui.colormap(
        "lte_colormap",
        label="Colormap",
        default="Plasma",
        options=COLORMAPS,
        group="Spectrogram display",
    )
    dbfs_min, dbfs_max = ui.limits(
        "lte_dbfs_limits",
        label="dBFS scale",
        default=(DBFS_MIN, DBFS_MAX),
        minimum=-120.0,
        maximum=0.0,
        step=1.0,
        group="Spectrogram display",
    )
    requested_fft_size = ui.select(
        "lte_fft_size",
        label="Fast-time FFT size (samples)",
        default=4096,
        options=FFT_SIZES,
        group="Spectrogram processing",
    )
    fft_window = ui.select(
        "lte_fft_window",
        label="Fast-time window",
        default="Hann",
        options=FFT_WINDOWS,
        group="Spectrogram processing",
    )
    overlap_percent = ui.select(
        "lte_overlap_percent",
        label="Slow-time overlap (%)",
        default=50,
        options=OVERLAPS,
        group="Spectrogram processing",
    )
    maximum_time_bins = ui.number(
        "lte_maximum_time_bins",
        label="Maximum slow-time bins",
        default=200,
        minimum=25,
        maximum=500,
        step=25,
        group="Spectrogram processing",
    )
    samples = data.samples[0]
    fft_size = min(int(requested_fft_size), samples.size)
    hop = max(1, round(fft_size * (1 - int(overlap_percent) / 100)))
    available_starts = np.arange(0, max(1, samples.size - fft_size + 1), hop)
    starts = available_starts[
        np.linspace(0, available_starts.size - 1, min(int(maximum_time_bins), available_starts.size), dtype=int)
    ]
    rows = np.asarray([samples[start : start + fft_size] for start in starts])
    taper = {
        "Hann": np.hanning,
        "Hamming": np.hamming,
        "Blackman": np.blackman,
        "Rectangular": np.ones,
    }[str(fft_window)](fft_size)
    spectra = np.fft.fftshift(np.fft.fft(rows * taper, axis=1), axes=1)
    power = np.abs(spectra / max(np.sum(taper), 1)) ** 2
    waterfall_db = 10 * np.log10(np.maximum(power, 1e-12))
    average_db = 10 * np.log10(np.maximum(np.mean(power, axis=0), 1e-12))

    captures = data.recording.metadata.get("captures", [])
    center_hz = float(captures[0].get("core:frequency", 0.0)) if captures else 0.0
    frequency_mhz = (center_hz + np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / data.sample_rate))) / 1e6
    time_ms = (data.start_sample + starts + fft_size / 2) / data.sample_rate * 1e3

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=(0.1, 0.9),
        vertical_spacing=0.04,
    )
    figure.add_trace(
        go.Scatter(x=frequency_mhz, y=average_db, mode="lines", line={"color": "#087e8b"}, showlegend=False),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            x=frequency_mhz,
            y=time_ms,
            z=waterfall_db,
            zmin=dbfs_min,
            zmax=dbfs_max,
            colorscale=colormap,
            colorbar={"title": "dBFS", "tickformat": ".1f"},
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text="PSD (dBFS)",
        range=[dbfs_min, dbfs_max],
        autorange=False,
        tickformat=".1f",
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text="Recording time (ms)", tickformat="07.2f", row=2, col=1)
    figure.update_xaxes(title_text="RF frequency (MHz)", tickformat="07.2f", row=2, col=1)
    figure.update_layout(uirevision=f"lte-spectrum:{data.recording.metadata_path.name}")

    ui.stat("Center frequency", f"{center_hz / 1e6:g} MHz")
    ui.stat("Sample rate", f"{data.sample_rate / 1e6:g} MS/s")
    ui.stat("Displayed samples", f"{samples.size:,}")
    with ui.tab("Spectrum + waterfall"):
        ui.plot(style_figure(figure, ui, "LTE spectrum"), key="lte-spectrum")


def create_workspace(config=None):
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data"))
    filename = str(values.get("filename", "*.sigmf-meta"))
    return AnalysisWorkspace(
        identifier=str(values.get("id", "lte-recordings")),
        name=str(values.get("name", "LTE Recordings")),
        description="Windowed mode: drag or resize an interval over sliding-median power and inspect its LTE time-frequency plot.",
        source=sigmf_source(root, filename, recursive=True),
        delivery=WindowedLteDelivery(),
        analyze=analyze,
        category="spectrum monitoring",
        tags=("windowed", "single-channel", "LTE", "spectrogram", "waterfall"),
    )
