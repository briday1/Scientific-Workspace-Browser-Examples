"""Pass-through analysis contract for already processed acoustic products."""

from .models import StoredEventResults


def process(event: StoredEventResults, settings: None) -> StoredEventResults:
    """The source already contains post-processed products, so preserve them."""
    return event


__all__ = ["process"]
