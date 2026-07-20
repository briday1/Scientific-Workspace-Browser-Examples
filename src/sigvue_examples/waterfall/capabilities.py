"""Select optional SigMF capabilities for file and collection sources."""

from ..io.sigmf.capabilities import SigMFExporter, WaterfallSigMFAnnotator
from .domain import GroupedWaterfallSigMFAnnotator


def waterfall_capabilities(is_collection: bool):
    if is_collection:
        return GroupedWaterfallSigMFAnnotator(), None
    return (
        WaterfallSigMFAnnotator("waterfall-spectrum", "waterfall_annotation_region_color"),
        SigMFExporter(),
    )
