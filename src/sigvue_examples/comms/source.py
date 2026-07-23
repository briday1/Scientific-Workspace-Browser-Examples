"""Bind the shared SigMF reader to communications dataset discovery."""

from pathlib import Path

from sigvue.plugin import DirectorySource

from ..io.sigmf import describe_recording, load_recording


def recording_source(root: Path, pattern: str = "*.sigmf-meta") -> DirectorySource:
    return DirectorySource(
        root,
        pattern=pattern,
        loader=lambda path: load_recording(path, sample_rate_fallback=1.0),
        describe=describe_recording,
        recursive=True,
    )
