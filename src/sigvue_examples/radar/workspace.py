"""LFM workspace for standard SigMF multi-stream captures."""

from pathlib import Path

from sigvue.helpers import WorkspaceConfig
from sigvue.plugin import Workspace

from ..plugins import (
    CallableAnalysis,
    CallablePresentation,
    SIGNAL_DISCOVERY_COLUMNS,
)
from .analysis import analyze_lfm, configure_lfm
from .capabilities import LfmAnnotator, LfmExporter
from .delivery import BufferedDelivery
from .presentation import present_lfm
from .source import collection_source


def create_workspace(config=None) -> Workspace:
    values = WorkspaceConfig(config)
    root = values.path("data_root", Path.cwd() / "data/lfm-sigmf")
    source = collection_source(
        root,
        calibration_dbm=values.floating("calibration_dbm", -20.0),
        ota_prf_hz=values.floating("ota_prf_hz", 1_000.0),
        ota_pulse_width_seconds=values.floating(
            "ota_pulse_width_seconds",
            50e-6,
        ),
    )
    return Workspace(
        identifier="lfm-sigmf",
        name="LFM SigMF View",
        delivery=BufferedDelivery(),
        description="Inspect standard multi-stream SigMF captures with the shared buffered LFM analysis pipeline.",
        source=source,
        annotator=LfmAnnotator(),
        exporter=LfmExporter(),
        analysis=CallableAnalysis(analyze_lfm, configure_lfm),
        presentation=CallablePresentation(present_lfm),
        lazy_views=True,
        category="signal analysis",
        tags=("live", "multi-channel", "SigMF", "LFM", "capture"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )


__all__ = ["create_workspace"]
