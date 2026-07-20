"""Pass-through analysis contract for already processed acoustic products."""

from sigvue.plugin import Analysis

from .domain import StoredEventResults, process


class EventAnalysis(Analysis[StoredEventResults, None, StoredEventResults]):
    def process(self, event: StoredEventResults, settings: None) -> StoredEventResults:
        return process(event, settings)


__all__ = ["EventAnalysis"]
