"""Plotly spectrum/waterfall presentation contract."""

from sigvue.plugin import Presentation, ViewContext

from .domain import WaterfallProducts, present_waterfall


class WaterfallPresentation(Presentation[WaterfallProducts]):
    def present(self, products: WaterfallProducts, ui: ViewContext) -> None:
        present_waterfall(products, ui)


__all__ = ["WaterfallPresentation"]
