"""Stored waveform and spectrum presentation contract."""

from sigvue.plugin import Presentation, ViewContext

from .domain import StoredEventResults, present


class EventPresentation(Presentation[StoredEventResults]):
    def present(self, event: StoredEventResults, ui: ViewContext) -> None:
        present(event, ui)


__all__ = ["EventPresentation"]
