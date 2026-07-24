#!/usr/bin/env python3
"""Download an EarthScope recording of the 2018 Anchorage earthquake."""

from __future__ import annotations

import argparse
from pathlib import Path
import ssl
import sys

import certifi
import numpy as np
from sigvue.helpers import RemoteFile, download_file
from sigvue_examples.plugins.sigmf import write_sigmf_recording


QUERY_URL = (
    "https://service.earthscope.org/fdsnws/dataselect/1/query?"
    "net=IU&sta=COLA&loc=00&cha=BHZ&"
    "start=2018-11-30T17:28:00&end=2018-11-30T17:43:00&"
    "format=geocsv.inline&nodata=404"
)
REMOTE = RemoteFile(
    url=QUERY_URL,
    filename="IU_COLA_00_BHZ_20181130T172800_15min.csv",
    size=1_318_486,
    checksum="sha256:9894bd297cacb681ad917e30316d469531d0e34f9551803f669f382d2d83566f",
)


def read_geocsv(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    metadata: dict[str, str] = {}
    samples: list[int] = []
    in_values = False
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("# ") and ":" in line:
            key, value = line[2:].split(":", 1)
            metadata.setdefault(key.strip(), value.strip())
            in_values = False
        elif line == "Time, Sample":
            in_values = True
        elif in_values and line:
            samples.append(int(line.rsplit(",", 1)[1]))
    if not samples:
        raise ValueError(f"{Path(path).name} contains no waveform samples")
    return np.asarray(samples, dtype=np.float64), metadata


def download_earthquake_seismic(output: str | Path) -> Path:
    root = Path(output)
    csv_path = download_file(
        REMOTE,
        root / "source-geocsv",
        user_agent="Sigvue-Examples/0.3",
        tls_context=ssl.create_default_context(cafile=certifi.where()),
    )
    counts, fields = read_geocsv(csv_path)
    scale = float(fields["scale_factor"])
    velocity = counts / scale
    metadata, _ = write_sigmf_recording(
        root,
        "anchorage-2018-IU-COLA-BHZ",
        velocity.astype(np.complex64),
        float(fields["sample_rate_hz"]),
        description="2018 M7.1 Anchorage earthquake · IU COLA BHZ",
        global_metadata={
            "core:datetime": fields["start_time"],
            "core:author": "EarthScope Consortium / IU Global Seismographic Network",
            "seismic:event": "2018 M7.1 Anchorage earthquake",
            "seismic:station": fields["SID"],
            "seismic:instrument": fields["instrument"],
            "seismic:latitude_deg": float(fields["latitude_deg"]),
            "seismic:longitude_deg": float(fields["longitude_deg"]),
            "seismic:elevation_m": float(fields["elevation_m"]),
            "seismic:depth_m": float(fields["depth_m"]),
            "seismic:azimuth_deg": float(fields["azimuth_deg"]),
            "seismic:dip_deg": float(fields["dip_deg"]),
            "seismic:units": fields["scale_units"],
            "seismic:source_url": QUERY_URL,
        },
        annotations=({
            "core:sample_start": round(
                89.0 * float(fields["sample_rate_hz"])
            ),
            "core:label": "M7.1 origin time",
            "core:comment": "USGS origin time 2018-11-30 17:29:29 UTC.",
        },),
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/earthquake-seismic"))
    args = parser.parse_args()
    path = download_earthquake_seismic(args.output)
    print(f"Ready: {path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
