"""Workspace assembly for stored acoustic-event results."""

from sigvue.helpers import WorkspaceConfig

from ..plugins import (
    CallableAnalysis,
    CallableDelivery,
    CallablePresentation,
    SIGNAL_DISCOVERY_COLUMNS,
)
from sigvue.plugin import Workspace

from .analysis import process
from .delivery import prepare
from .presentation import present
from .source import event_source


def create_workspace(config=None) -> Workspace:
    values = WorkspaceConfig(config)
    root = values.path(
        "data_root",
        "data/acoustic-events-segmented",
    )
    return Workspace(
        identifier="acoustic-events-segmented",
        name="Acoustic Event Review",
        description="Segmented mode: navigate irregular precomputed acoustic events and display stored results without reprocessing raw data.",
        source=event_source(root),
        delivery=CallableDelivery(prepare),
        analysis=CallableAnalysis(process),
        presentation=CallablePresentation(present),
        lazy_views=True,
        category="acoustic monitoring",
        tags=("segmented", "irregular events", "precomputed", "display-only"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )
