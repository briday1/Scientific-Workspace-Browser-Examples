"""STFT waterfall analysis contract."""

from sigvue.plugin import Analysis, ParameterContext

from .domain import (
    WaterfallWindow,
    WaterfallProducts,
    WaterfallSettings,
    _waterfall_spectrogram,
    configure_waterfall,
    process_waterfall,
)


class WaterfallAnalysis(Analysis[WaterfallWindow, WaterfallSettings, WaterfallProducts]):
    def configure(self, data: WaterfallWindow, ui: ParameterContext) -> WaterfallSettings:
        return configure_waterfall(data, ui)

    def process(
        self,
        data: WaterfallWindow,
        settings: WaterfallSettings | None,
    ) -> WaterfallProducts:
        if settings is None:
            raise RuntimeError("Waterfall analysis requires configured settings")
        return process_waterfall(data, settings)

__all__ = [
    "WaterfallProducts",
    "WaterfallAnalysis",
    "WaterfallSettings",
    "_waterfall_spectrogram",
    "configure_waterfall",
    "process_waterfall",
]
