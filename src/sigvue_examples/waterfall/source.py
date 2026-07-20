"""Choose file or collection discovery for the shared waterfall pipeline."""

from pathlib import Path

from .domain import SigMFCollectionSource, _recording_source


def waterfall_source(root: Path, filename: str, source_type: str):
    is_collection = source_type == "collection" or "sigmf-collection" in filename
    source = (
        SigMFCollectionSource(root, filename)
        if is_collection
        else _recording_source(root, filename, recursive=True)
    )
    return source, is_collection
