"""Annotation and export implementations for LFM collections."""

from pathlib import Path
from uuid import uuid4

from sigvue.plugin import Annotation, AnnotationRequest, Annotator, Exporter, ExportRequest

from ..plugins.sigmf import (
    SAMPLE_EXPORT_FORMATS,
    SAMPLE_EXPORT_SCOPES,
    add_sigmf_annotation,
    load_sigmf_recording,
    read_sigmf_annotations,
    waterfall_annotation_fields,
    write_array_bundle_export,
)

from .models import LfmCollection, LfmInput


class LfmAnnotator(Annotator[LfmCollection, LfmInput]):
    """Store matching standard SigMF annotations on every OTA member."""

    timeline_color_control = "lfm_annotation_region_color"

    @property
    def fields(self):
        return waterfall_annotation_fields(
            "waterfall-domain-1",
            time_scale=1.0,
            frequency_scale=1.0,
            time_offset_source="playback",
        )

    def discover(self, collection: LfmCollection):
        return read_sigmf_annotations(
            load_sigmf_recording(
                collection.members["ota"][0].metadata_path,
            )
        )

    def annotate(self, collection: LfmCollection, delivered: LfmInput, request: AnnotationRequest) -> Annotation:
        try:
            start_seconds = float(request.values["start_seconds"])
            stop_seconds = float(request.values["stop_seconds"])
            frequency_lower_hz = float(request.values["frequency_lower_hz"])
            frequency_upper_hz = float(request.values["frequency_upper_hz"])
        except (KeyError, ValueError) as error:
            raise ValueError("Waterfall annotation bounds must be numeric") from error
        if start_seconds < 0 or stop_seconds <= start_seconds:
            raise ValueError("Annotation stop time must be after its non-negative start time")
        available = collection.sample_count("ota")
        start_sample = min(available, round(start_seconds * collection.sample_rate))
        stop_sample = min(available, round(stop_seconds * collection.sample_rate))
        if stop_sample <= start_sample:
            raise ValueError("Annotation time bounds do not contain any recording samples")
        identifier = str(uuid4())
        result = None
        for member in collection.members["ota"]:
            result = add_sigmf_annotation(
                load_sigmf_recording(member.metadata_path),
                start_sample,
                stop_sample - start_sample,
                request,
                identifier=identifier,
                frequency_lower_hz=frequency_lower_hz,
                frequency_upper_hz=frequency_upper_hz,
                generator="Sigvue Examples",
            )
        assert result is not None
        return result

class LfmExporter(Exporter[LfmCollection, LfmInput]):
    """Serialize either the delivered OTA window or every collection member."""

    @property
    def scopes(self):
        return SAMPLE_EXPORT_SCOPES

    @property
    def formats(self):
        return SAMPLE_EXPORT_FORMATS

    def export(self, collection: LfmCollection, delivered: LfmInput, request: ExportRequest, directory: Path) -> Path:
        stem = collection.collection_path.stem if collection.collection_path else "lfm-collection"
        if request.scope == "buffer":
            start = delivered.start_sample
            arrays = {
                "calibration": delivered.calibration_counts,
                "terminated_noise": delivered.noise_counts,
                "ota": delivered.ota_counts,
            }
        else:
            start = 0
            arrays = {role.replace("-", "_"): collection.read(role) for role in collection.members}
        ota_count = arrays["ota"].shape[-1]
        return write_array_bundle_export(
            directory,
            stem=stem,
            arrays=arrays,
            sample_rate=collection.sample_rate,
            start_sample=start,
            sample_count=ota_count,
            scope=request.scope,
            format=request.format,
            metadata={
                "calibration_dbm": collection.calibration_dbm,
                "adc_bits": collection.adc_bits,
                "ota_prf_hz": collection.ota_prf_hz,
            },
            control_values=request.control_values,
        )


__all__ = ["LfmAnnotator", "LfmExporter"]
