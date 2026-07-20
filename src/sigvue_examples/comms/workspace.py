"""Workspace assembly for the digital communications pipeline."""

from pathlib import Path

from sigvue.plugin import Workspace

from ..io.sigmf.capabilities import SIGNAL_DISCOVERY_COLUMNS, SigMFAnnotator, SigMFExporter
from .analysis import CommsAnalysis
from .delivery import WindowedCommsDelivery
from .plots import CommsPresentation
from .source import recording_source


def create_workspace(config=None) -> Workspace:
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data/comms"))
    return Workspace(
        identifier="digital-comms",
        name="Digital Communications",
        description="Windowed mode: compare file-backed QPSK and 16-QAM recordings with constellation and eye-diagram views.",
        source=recording_source(root),
        delivery=WindowedCommsDelivery(),
        annotator=SigMFAnnotator(),
        exporter=SigMFExporter(),
        analysis=CommsAnalysis(),
        presentation=CommsPresentation(),
        category="digital communications",
        tags=("windowed", "qpsk", "16-qam", "constellation", "eye diagram"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )
