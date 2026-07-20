"""Calibration and radar time/frequency analysis contract."""

from sigvue.plugin import Analysis, ParameterContext

from .domain import LfmAnalysisProducts, LfmInput, LfmSettings, configure_lfm, process_lfm


class LfmAnalysis(Analysis[LfmInput, LfmSettings, LfmAnalysisProducts]):
    def configure(self, data: LfmInput, ui: ParameterContext) -> LfmSettings:
        return configure_lfm(data, ui)

    def process(
        self,
        data: LfmInput,
        settings: LfmSettings | None,
    ) -> LfmAnalysisProducts:
        if settings is None:
            raise RuntimeError("LFM analysis requires configured settings")
        return process_lfm(data, settings)


__all__ = ["LfmAnalysis", "LfmAnalysisProducts", "LfmSettings"]
