"""Assembly of the EarthScope earthquake workspace."""

from sigvue.helpers import WorkspaceConfig
from sigvue.plugin import Workspace

from ..plugins import CallableAnalysis, CallablePresentation
from ..plugins.sigmf import SIGMF_DISCOVERY_COLUMNS, WindowedSigMFDelivery, sigmf_source
from .analysis import configure, process
from .presentation import present


def create_workspace(config=None) -> Workspace:
    root = WorkspaceConfig(config).path("data_root", "data/earthquake-seismic")
    return Workspace(
        identifier="earthquake-seismic",
        name="Earthquake Seismology",
        description="EarthScope recording of the 2018 M7.1 Anchorage earthquake.",
        source=sigmf_source(root, tags=("EarthScope", "earthquake", "seismology", "real data")),
        delivery=WindowedSigMFDelivery(
            default_window=900,
            minimum_window=30,
            step=10,
            overview_bins=300,
            overview_label="Mean ground-motion power",
            cache_key="seismic-power",
        ),
        analysis=CallableAnalysis(process, configure),
        presentation=CallablePresentation(present),
        lazy_views=True,
        category="geophysics",
        tags=("EarthScope", "earthquake", "seismology", "real data"),
        discovery_columns=SIGMF_DISCOVERY_COLUMNS,
    )
