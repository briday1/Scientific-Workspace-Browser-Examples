"""Assembly of the NOAA passive-acoustics workspace."""

from sigvue.helpers import WorkspaceConfig
from sigvue.plugin import Workspace

from ..plugins import CallableAnalysis, CallablePresentation
from ..plugins.sigmf import SIGMF_DISCOVERY_COLUMNS, WindowedSigMFDelivery, sigmf_source
from .analysis import configure, process
from .presentation import present


def create_workspace(config=None) -> Workspace:
    values = WorkspaceConfig(config)
    root = values.path("data_root", "data/passive-acoustics")
    return Workspace(
        identifier="noaa-passive-acoustics",
        name="NOAA Passive Acoustics",
        description="Real SanctSound hydrophone clips with progressive spectrograms.",
        source=sigmf_source(root, tags=("NOAA", "SanctSound", "hydrophone", "real data")),
        delivery=WindowedSigMFDelivery(
            default_window=15.0,
            minimum_window=1.0,
            step=0.5,
            overview_bins=240,
            overview_label="Mean acoustic power (dBFS)",
            time_unit="s",
            cache_key="passive-acoustic-power",
        ),
        analysis=CallableAnalysis(process, configure),
        presentation=CallablePresentation(present),
        lazy_views=True,
        category="ocean acoustics",
        tags=("NOAA", "SanctSound", "hydrophone", "blue whale", "real data"),
        discovery_columns=SIGMF_DISCOVERY_COLUMNS,
    )


__all__ = ["create_workspace"]
