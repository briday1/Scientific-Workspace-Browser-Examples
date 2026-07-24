"""Framework assembly for the windowed communications pipeline."""

from sigvue.helpers import WorkspaceConfig

from ..plugins import CallableAnalysis, CallablePresentation
from ..plugins.sigmf import (
    SIGMF_DISCOVERY_COLUMNS,
    SigMFAnnotator,
    SigMFExporter,
    WindowedSigMFDelivery,
    sigmf_source,
)
from sigvue.plugin import Workspace

from .analysis import process
from .presentation import present


def create_workspace(config=None) -> Workspace:
    values = WorkspaceConfig(config)
    root = values.path("data_root", "data/comms")
    return Workspace(
        identifier="synthetic-comms",
        name="Synthetic Communications",
        description="Windowed constellation and eye-diagram analysis for generated QPSK, 16-QAM, and 64-QAM recordings.",
        source=sigmf_source(
            root,
            pattern=values.string("filename", "synthetic-*.sigmf-meta"),
        ),
        annotator=SigMFAnnotator(generator="Sigvue Examples"),
        exporter=SigMFExporter(),
        delivery=WindowedSigMFDelivery(
            default_window=0.012,
            minimum_window=0.002,
            step=0.001,
            overview_bins=240,
            overview_channel=0,
            time_unit="ms",
            cache_key="comms-power-overview",
        ),
        analysis=CallableAnalysis(process),
        presentation=CallablePresentation(present),
        lazy_views=True,
        category="digital communications",
        tags=("windowed", "synthetic", "SigMF", "QPSK", "16-QAM", "64-QAM"),
        discovery_columns=SIGMF_DISCOVERY_COLUMNS,
    )
