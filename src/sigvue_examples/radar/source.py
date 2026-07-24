"""Adapter from standard SigMF multi-stream collections to the LFM pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from sigvue.plugin import DataResource, DirectorySource

from ..plugins.sigmf import (
    load_metadata,
    load_sigmf_recording,
    sigmf_discovery_summary,
)

from .models import CollectionMember, LfmCollection


def _streams(path: Path) -> tuple[dict[str, object], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    streams = tuple(payload.get("collection", {}).get("core:streams", ()))
    if not streams:
        raise ValueError(f"{path.name} does not define any core:streams")
    return streams


def describe_collection(path: Path) -> DataResource:
    streams = _streams(path)
    metadata_paths = [path.parent / str(stream["name"]) for stream in streams]
    metadata = load_metadata(metadata_paths[0])
    ota_count = sum(stream.get("lfm:role") == "ota" for stream in streams)
    channel_count = ota_count or len(metadata_paths)
    collection = json.loads(path.read_text(encoding="utf-8"))["collection"]
    return DataResource(
        path.stem,
        str(collection.get("core:description") or path.parent.name.replace("_", " ")),
        source=path,
        tags=("sigmf-collection", "sc16", f"{channel_count}-channel"),
        summary=sigmf_discovery_summary(metadata),
    )


def read_collection(
    path: Path,
    *,
    calibration_dbm: float = -20.0,
    ota_prf_hz: float = 1_000.0,
    ota_pulse_width_seconds: float = 50e-6,
) -> LfmCollection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    collection_metadata = payload["collection"]
    streams = _streams(path)
    grouped: dict[str, list[CollectionMember]] = {}
    sample_rate = None
    has_roles = all(stream.get("lfm:role") is not None for stream in streams)
    for fallback_channel, stream in enumerate(streams, start=1):
        metadata_path = path.parent / str(stream["name"])
        recording = load_sigmf_recording(metadata_path)
        if recording.datatype not in {"sc16_le", "ci16_le"}:
            raise ValueError(f"{metadata_path.name} must be single-channel sc16_le")
        if recording.channel_count != 1:
            raise ValueError(f"{metadata_path.name} must contain one channel")
        rate = recording.sample_rate
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(f"{metadata_path.name} has a different sample rate")
        role = str(stream.get("lfm:role", "ota"))
        channel = int(stream.get("lfm:channel", fallback_channel))
        grouped.setdefault(role, []).append(
            CollectionMember(
                role,
                channel,
                recording.metadata_path,
                recording.data_path,
                float(
                    stream.get(
                        "lfm:duration_seconds",
                        recording.duration_seconds,
                    )
                ),
            )
        )

    if has_roles:
        required = {"calibration", "terminated-noise", "ota"}
        if set(grouped) != required:
            raise ValueError(f"LFM SigMF collection must define exactly {sorted(required)} roles")
        roles = {
            role: tuple(sorted(records, key=lambda member: member.channel))
            for role, records in grouped.items()
        }
    else:
        # Role-free field captures remain viewable with the shared pipeline.
        records = tuple(sorted(grouped["ota"], key=lambda member: member.channel))
        roles = {"calibration": records, "terminated-noise": records, "ota": records}
    return LfmCollection(
        float(sample_rate),
        float(collection_metadata.get("lfm:calibration_dbm", calibration_dbm)),
        16,
        roles,
        float(collection_metadata.get("lfm:ota_prf_hz", ota_prf_hz)),
        float(collection_metadata.get("lfm:ota_pulse_width_seconds", ota_pulse_width_seconds)),
        path,
    )


def collection_source(
    root: Path,
    *,
    calibration_dbm: float = -20.0,
    ota_prf_hz: float = 1_000.0,
    ota_pulse_width_seconds: float = 50e-6,
) -> DirectorySource:
    def load(path: Path) -> LfmCollection:
        return read_collection(
            path,
            calibration_dbm=calibration_dbm,
            ota_prf_hz=ota_prf_hz,
            ota_pulse_width_seconds=ota_pulse_width_seconds,
        )

    return DirectorySource(
        root,
        pattern="*.sigmf-collection",
        loader=load,
        describe=describe_collection,
        recursive=True,
    )


__all__ = [
    "collection_source",
    "describe_collection",
    "read_collection",
]
