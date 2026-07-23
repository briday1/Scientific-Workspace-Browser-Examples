"""Live/seek and whole-file UI delivery policies for LFM collections."""

from sigvue.plugin import Delivery, DeliveryContext, PlaybackMode

from ..io.sigmf.capabilities import read_sigmf_annotations
from ..io.sigmf.recording import load_recording
from .domain import LfmCollection, LfmInput


class BufferedDelivery(Delivery[LfmCollection, LfmInput]):
    """Framework policy for playback: deliver one requested OTA window."""

    def __init__(self, *, playback_mode: PlaybackMode = "live") -> None:
        if playback_mode not in {"seek", "live"}:
            raise ValueError("Buffered playback mode must be 'seek' or 'live'")
        self.playback_mode = playback_mode

    def prepare(self, collection: LfmCollection, ui: DeliveryContext) -> LfmInput:
        default_pri = 1 / collection.ota_prf_hz
        available = collection.sample_count("ota")
        recording_seconds = available / collection.sample_rate
        default_buffer_seconds = min(0.02, max(default_pri, recording_seconds / 4))
        buffer_seconds = ui.number(
            "buffer_seconds",
            default=default_buffer_seconds,
            minimum=min(default_pri, recording_seconds),
            maximum=max(min(default_pri, recording_seconds), min(0.1, recording_seconds)),
            step=default_pri,
        )
        processing_pri_seconds = ui.number(
            "processing_pri_seconds",
            label="Processing PRI (s)",
            default=default_pri,
            minimum=8 / collection.sample_rate,
            maximum=1.0,
            step=default_pri / 10,
        )
        seek_seconds = ui.number("seek_seconds", default=0.01, minimum=0.001, step=0.001)
        refresh_seconds = ui.number("refresh_seconds", default=1.0, minimum=0.1, step=0.1)
        size = min(available, max(1, round(buffer_seconds * collection.sample_rate)))
        pri = min(size, max(8, round(collection.sample_rate * processing_pri_seconds)))
        duration = available / collection.sample_rate
        time = ui.playback(
            mode=self.playback_mode,
            duration=duration,
            step=seek_seconds,
            refresh_interval=refresh_seconds,
            loop=False,
        )
        start = min(round(time * collection.sample_rate), available - size)
        return _input(collection, start=start, count=size, pri=pri, ui=ui)

class WholeFileDelivery(Delivery[LfmCollection, LfmInput]):
    """Framework policy for batch mode: deliver the complete OTA member files."""

    def __init__(self, *, default_processing_pri_seconds: float | None = None) -> None:
        self.default_processing_pri_seconds = default_processing_pri_seconds

    def prepare(self, collection: LfmCollection, ui: DeliveryContext) -> LfmInput:
        ui.playback(mode="static")
        default_pri_seconds = self.default_processing_pri_seconds or 1 / collection.ota_prf_hz
        processing_pri_seconds = ui.number(
            "processing_pri_seconds",
            label="Processing PRI (s)",
            default=default_pri_seconds,
            minimum=8 / collection.sample_rate,
            maximum=1.0,
            step=default_pri_seconds / 10,
        )
        pri = max(8, round(collection.sample_rate * processing_pri_seconds))
        return _input(collection, start=0, count=collection.sample_count("ota"), pri=pri, ui=ui)

def _input(collection: LfmCollection, *, start: int, count: int, pri: int, ui: DeliveryContext) -> LfmInput:
    calibration = ui.once("lfm-calibration-counts", lambda: collection.read("calibration"))
    noise = (
        calibration
        if _same_recordings(collection, "calibration", "terminated-noise")
        else ui.once("lfm-noise-counts", lambda: collection.read("terminated-noise"))
    )
    ota = (
        calibration
        if start == 0
        and count == collection.sample_count("ota")
        and _same_recordings(collection, "calibration", "ota")
        else collection.read("ota", start, count)
    )
    annotation_path = collection.members["ota"][0].metadata_path
    current_annotations = read_sigmf_annotations(load_recording(annotation_path)) if annotation_path.is_file() else ()
    return LfmInput(
        sample_rate=collection.sample_rate,
        calibration_dbm=collection.calibration_dbm,
        adc_bits=collection.adc_bits,
        pri_samples=pri,
        start_sample=start,
        calibration_counts=calibration,
        noise_counts=noise,
        ota_counts=ota,
        annotations=current_annotations,
    )


def _same_recordings(collection: LfmCollection, left: str, right: str) -> bool:
    return tuple(
        (member.metadata_path, member.data_path) for member in collection.members[left]
    ) == tuple(
        (member.metadata_path, member.data_path) for member in collection.members[right]
    )


__all__ = ["BufferedDelivery", "LfmInput", "WholeFileDelivery"]
