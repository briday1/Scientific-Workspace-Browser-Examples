"""File discovery and opening for the communications pipeline."""

from pathlib import Path

from sigvue.plugin import DirectorySource

from ..io.sigmf.recording import load_recording
from .domain import _describe_recording


def recording_source(root: Path) -> DirectorySource:
    return DirectorySource(
        root,
        pattern="*.sigmf-meta",
        loader=lambda path: load_recording(path, sample_rate_fallback=1.0),
        describe=_describe_recording,
    )
