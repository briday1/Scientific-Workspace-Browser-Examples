"""Stored-result discovery and JSON loading."""

from pathlib import Path

from sigvue.plugin import DirectorySource

from .domain import AcousticEventCollection, describe_collection, load_collection


def event_source(root: Path) -> DirectorySource:
    return DirectorySource(
        root,
        pattern="acoustic-events.json",
        loader=load_collection,
        describe=describe_collection,
    )


__all__ = ["AcousticEventCollection", "event_source"]
