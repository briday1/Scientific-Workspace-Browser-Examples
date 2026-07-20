"""Constellation and eye-diagram presentation contract."""

from sigvue.plugin import Presentation, ViewContext

from .domain import CommsProducts, present


class CommsPresentation(Presentation[CommsProducts]):
    def present(self, products: CommsProducts, ui: ViewContext) -> None:
        present(products, ui)


__all__ = ["CommsPresentation"]
