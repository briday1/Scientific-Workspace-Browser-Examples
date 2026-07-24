#!/usr/bin/env python3
"""Download two compact NOAA SanctSound blue-whale clips and convert to SigMF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import ssl
import sys
import wave

import certifi
import numpy as np
from sigvue.helpers import RemoteFile, download_file
from sigvue_examples.plugins.sigmf import write_sigmf_recording


BUCKET_OBJECT = (
    "https://storage.googleapis.com/download/storage/v1/b/"
    "noaa-passive-bioacoustic/o/"
    "sanctsound%2Fproducts%2Fsound_clips%2Fmb01%2F"
    "sanctsound_mb01_04_sound_clips%2Fdata%2F"
)


@dataclass(frozen=True)
class Clip:
    filename: str
    generation: str
    size: int
    sha256: str
    event: str
    timestamp: str

    @property
    def remote(self) -> RemoteFile:
        return RemoteFile(
            url=f"{BUCKET_OBJECT}{self.filename}?generation={self.generation}&alt=media",
            filename=self.filename,
            size=self.size,
            checksum=f"sha256:{self.sha256}",
        )


CLIPS = (
    Clip(
        "SanctSound_MB01_04_bluewhaleAcall_20200124T031012Z_8xSpeed.wav",
        "1685132307389518",
        489_286,
        "e19b580555e6c340108563af84f52fffecf7f1f986c2fb7e601a3d26ff6fb5d9",
        "Blue whale A call",
        "2020-01-24T03:10:12Z",
    ),
    Clip(
        "SanctSound_MB01_04_bluewhaleBcall_20200118T042613Z_8xSpeed.wav",
        "1685132308671088",
        1_248_374,
        "f84b89aa2764a4d09cf78490edf318296026fac2e4cb422e9f720bc2d2e6cfae",
        "Blue whale B call",
        "2020-01-18T04:26:13Z",
    ),
)


def convert_clip(path: Path, output: Path, clip: Clip) -> tuple[Path, Path]:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f"{path.name} is not mono 16-bit PCM")
        sample_rate = stream.getframerate()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    normalized = samples.astype(np.float32) / 32768.0
    stem = path.stem
    return write_sigmf_recording(
        output,
        stem,
        normalized.astype(np.complex64),
        sample_rate,
        description=f"SanctSound MB01 · {clip.event} (8× playback)",
        global_metadata={
            "core:datetime": clip.timestamp,
            "core:author": "NOAA Office of National Marine Sanctuaries and U.S. Navy",
            "core:license": "https://doi.org/10.25921/saca-sp25",
            "audio:archive": "NOAA NCEI Passive Acoustic Data Archive",
            "audio:project": "SanctSound",
            "audio:site": "MB01",
            "audio:event": clip.event,
            "audio:playback": "8× original speed",
            "audio:source_url": clip.remote.url,
            "audio:citation": (
                "NOAA Office of National Marine Sanctuaries and U.S Navy. "
                "2020. SanctSound Raw Passive Acoustic Data."
            ),
        },
        annotations=({
            "core:sample_start": 0,
            "core:sample_count": int(samples.size),
            "core:label": clip.event,
            "core:comment": "NOAA-provided labeled sound clip; accelerated 8×.",
        },),
    )


def download_passive_acoustics(output: str | Path) -> tuple[Path, ...]:
    root = Path(output)
    downloads = root / "source-wav"
    tls = ssl.create_default_context(cafile=certifi.where())
    results = []
    for clip in CLIPS:
        wav_path = download_file(
            clip.remote,
            downloads,
            user_agent="Sigvue-Examples/0.3",
            tls_context=tls,
        )
        metadata, _ = convert_clip(wav_path, root, clip)
        results.append(metadata)
    return tuple(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/passive-acoustics"))
    args = parser.parse_args()
    paths = download_passive_acoustics(args.output)
    print(f"Ready: {len(paths)} NOAA passive-acoustic clips in {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
