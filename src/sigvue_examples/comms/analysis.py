"""Communications signal processing contract."""

from sigvue.plugin import Analysis

from .domain import CommsProducts, CommsWindow, process


class CommsAnalysis(Analysis[CommsWindow, None, CommsProducts]):
    def process(self, data: CommsWindow, settings: None) -> CommsProducts:
        return process(data, settings)


__all__ = ["CommsAnalysis", "CommsProducts"]
