"""Presentation of EarthScope earthquake data."""

from sigvue.helpers import format_bytes
from sigvue.plugin import ViewContext

from .analysis import SeismicProducts
from .plots import spectrogram, spectrum, waveform


def present(products: SeismicProducts, ui: ViewContext) -> None:
    metadata = products.window.recording.metadata["global"]
    ui.stat("Event", metadata["seismic:event"])
    ui.stat("Station", metadata["seismic:station"])
    ui.stat("Channel", "BHZ vertical")
    ui.stat("Sample rate", f"{products.window.recording.sample_rate:g} Hz")
    ui.stat("Window memory", format_bytes(products.window.buffer_nbytes))
    with ui.tab("Waveform"):
        ui.plot(lambda: waveform(products, ui.theme), key="seismic-waveform")
    with ui.tab("Spectrogram"):
        ui.plot(
            lambda: spectrogram(
                products, ui.theme, ui.plot_viewport("seismic-spectrogram")
            ),
            key="seismic-spectrogram",
            axis_navigation="bounded",
        )
    with ui.tab("Spectrum"):
        ui.plot(lambda: spectrum(products, ui.theme), key="seismic-spectrum")
    with ui.tab("Metadata"):
        ui.table(
            lambda: [
                {"Field": key.removeprefix("seismic:"), "Value": value}
                for key, value in metadata.items()
                if key.startswith("seismic:")
            ],
            key="seismic-metadata",
        )
