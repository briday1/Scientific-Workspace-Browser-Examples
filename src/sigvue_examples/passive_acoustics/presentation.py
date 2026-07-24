"""Workspace presentation for NOAA SanctSound clips."""

from __future__ import annotations

from sigvue.helpers import format_bytes
from sigvue.plugin import ViewContext

from .analysis import AcousticProducts
from .plots import psd_figure, spectrogram_figure, waveform_figure


def present(products: AcousticProducts, ui: ViewContext) -> None:
    recording = products.window.recording
    metadata = recording.metadata["global"]
    render_width = int(ui.select(
        "acoustic_render_width", label="Spectrogram E/W pixels", default=1024,
        options=(256, 512, 1024, 2048), group="Raster rendering",
    ))
    render_height = int(ui.select(
        "acoustic_render_height", label="Spectrogram frequency pixels", default=512,
        options=(128, 256, 512, 1024), group="Raster rendering",
    ))
    ui.stat("Sound", metadata.get("audio:event", "Ocean sound"))
    ui.stat("Site", metadata.get("audio:site", "MB01"))
    ui.stat("Sample rate", f"{recording.sample_rate / 1e3:g} kHz")
    ui.stat("Duration", f"{recording.duration_seconds:.2f} s")
    ui.stat("Window memory", format_bytes(products.window.buffer_nbytes))
    with ui.tab("Soundscape"):
        ui.plot(
            lambda: spectrogram_figure(
                products,
                theme=ui.theme,
                viewport=ui.plot_viewport("passive-acoustic-spectrogram"),
                render_width=render_width,
                render_height=render_height,
            ),
            key="passive-acoustic-spectrogram",
            axis_navigation="bounded",
        )
    with ui.tab("Waveform"):
        ui.plot(lambda: waveform_figure(products, ui.theme), key="passive-acoustic-waveform")
    with ui.tab("Spectrum"):
        ui.plot(lambda: psd_figure(products, ui.theme), key="passive-acoustic-psd")
    with ui.tab("Metadata"):
        ui.table(
            lambda: [
                {"Field": key, "Value": value}
                for key, value in (
                    ("Archive", metadata.get("audio:archive")),
                    ("Project", metadata.get("audio:project")),
                    ("Site", metadata.get("audio:site")),
                    ("Detected sound", metadata.get("audio:event")),
                    ("Original timestamp", metadata.get("core:datetime")),
                    ("Playback", metadata.get("audio:playback")),
                    ("Source URL", metadata.get("audio:source_url")),
                    ("Citation", metadata.get("audio:citation")),
                )
            ],
            key="passive-acoustic-metadata",
        )
