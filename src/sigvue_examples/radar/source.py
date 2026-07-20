"""SigMF collection discovery for calibrated radar recordings."""

from pathlib import Path

from sigvue.plugin import DirectorySource

from .domain import describe_collection, read_collection


def collection_source(root: Path) -> DirectorySource:
    return DirectorySource(
        root,
        pattern="*.sigmf-collection",
        loader=read_collection,
        describe=describe_collection,
        recursive=True,
    )
