"""Radar calibration, time, frequency, and waterfall presentation contract."""

from sigvue.plugin import Presentation, ViewContext

from .domain import LfmAnalysisProducts, present_lfm


class LfmPresentation(Presentation[LfmAnalysisProducts]):
    def present(self, products: LfmAnalysisProducts, ui: ViewContext) -> None:
        present_lfm(products, ui)


__all__ = ["LfmPresentation"]
